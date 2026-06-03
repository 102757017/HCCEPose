#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并版 ONNX 导出脚本 (基于 HccePose 模型)
功能：
    - 导出 FP32 ONNX 模型
    - (可选) 基于测试数据集进行 INT8 静态量化
使用前请根据实际情况修改下方的用户配置区域
"""

import os
import json
import torch
import numpy as np

# 模型与数据相关导入
from HccePose.network_model import HccePose_BF_Net
from HccePose.bop_loader import bop_dataset, test_bop_dataset_back_front

# ONNX 相关导入
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process


# ==============================
# 公共函数：FP32 ONNX 导出
# ==============================
class HccePoseExportWrapper(torch.nn.Module):
    """导出专用封装：仅包含纯网络前向"""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def load_object_bbox_from_json(models_info_path: str, obj_id: int):
    """从 BOP 格式的 models_info.json 读取物体的包围盒参数"""
    with open(models_info_path, 'r') as f:
        models_info = json.load(f)
    obj_key = str(obj_id)
    if obj_key not in models_info:
        raise KeyError(f"物体 ID {obj_id} 不存在于 {models_info_path} 中")
    info = models_info[obj_key]
    min_xyz = torch.tensor([info['min_x'], info['min_y'], info['min_z']], dtype=torch.float32)
    size_xyz = torch.tensor([info['size_x'], info['size_y'], info['size_z']], dtype=torch.float32)
    return min_xyz, size_xyz


def export_hccepose_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    min_xyz: torch.Tensor = None,
    size_xyz: torch.Tensor = None,
    models_info_path: str = None,
    obj_id: int = None,
    efficientnet_key=None,
    input_size: int = 256,
    device: str = "cuda",
    opset_version: int = 17,
    dynamic_batch: bool = True,
):
    """
    导出 HccePose 模型为 FP32 ONNX 格式
    """
    # 验证并获取 min_xyz, size_xyz
    if min_xyz is None or size_xyz is None:
        if models_info_path is not None and obj_id is not None:
            min_xyz, size_xyz = load_object_bbox_from_json(models_info_path, obj_id)
        else:
            raise ValueError("必须提供 (min_xyz, size_xyz) 或 (models_info_path, obj_id) 两者之一")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    # 设备选择
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        device_torch = torch.device("cpu")
    else:
        device_torch = torch.device(device)

    min_xyz = min_xyz.to(device_torch)
    size_xyz = size_xyz.to(device_torch)

    # 构建模型并加载权重
    model = HccePose_BF_Net(
        efficientnet_key=efficientnet_key,
        min_xyz=min_xyz,
        size_xyz=size_xyz,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device_torch).eval()
    print(f"模型已加载: {checkpoint_path}")

    # 包装模型（仅网络主干）
    net_to_export = model.net if hasattr(model, "net") else model
    export_model = HccePoseExportWrapper(net_to_export).to(device_torch).eval()

    # 准备 dummy 输入
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device_torch)

    # 动态轴设置
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "inputs": {0: "batch_size"},
            "pred_mask_logits": {0: "batch_size"},
            "pred_front_back_code_logits": {0: "batch_size"},
        }

    # 导出 ONNX
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # 动态轴设置方式改变
    if dynamic_batch:
        dynamic_shapes = {
            "inputs": {0: torch.export.Dim("batch_size")},
            "pred_mask_logits": {0: torch.export.Dim("batch_size")},
            "pred_front_back_code_logits": {0: torch.export.Dim("batch_size")},
        }
    else:
        dynamic_shapes = None
    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            dummy_input,
            output_onnx_path,
            input_names=["inputs"],
            output_names=["pred_mask_logits", "pred_front_back_code_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,  #使用旧版导出方式
            verbose=False,
        )
    print(f"FP32 ONNX 模型已保存至: {output_onnx_path}")

    # 验证 ONNX 模型
    try:
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("FP32 ONNX 模型检查通过")
    except Exception as e:
        print(f"ONNX 模型验证失败: {e}")


# ==============================
# INT8 量化相关函数
# ==============================
def get_calibration_dataloader(
    dataset_path: str,
    obj_id: int,
    obj_path: str = None,
    train_folder_name: str = "train_pbr",
    padding_ratio: float = 1.5,
    ratio: float = 0.01,
    input_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0,
    local_rank: int = 0,
):
    """
    构建用于量化的校准数据 DataLoader
    返回 torch DataLoader，每个元素是预处理后的 RGB 图像 (float32, 范围 0~255)
    """
    bop_dataset_item = bop_dataset(dataset_path, local_rank=local_rank)
    test_dataset = test_bop_dataset_back_front(
        bop_dataset_item,
        train_folder_name,
        padding_ratio=padding_ratio,
        ratio=ratio
    )

    if obj_path is None:
        obj_path = os.path.join(dataset_path, 'models', f'obj_{obj_id:06d}.ply')
        if not os.path.exists(obj_path):
            alt_path = os.path.join(dataset_path, 'models', f'obj_{obj_id:02d}.ply')
            if os.path.exists(alt_path):
                obj_path = alt_path
            else:
                raise FileNotFoundError(f"找不到物体模型文件: {obj_path} 或 {alt_path}")
    test_dataset.update_obj_id(obj_id, obj_path)

    class CalibWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            rgb_c = self.ds[idx][0]
            if not torch.is_floating_point(rgb_c):
                rgb_c = rgb_c.float()
            return rgb_c

    calib_dataset = CalibWrapper(test_dataset)
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return calib_loader


class CalibrationDataReader(CalibrationDataReader):
    """ONNX Runtime 静态量化所需的数据读取器"""
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)

    def get_next(self):
        try:
            batch = next(self.iter)
            # 注意：输入名必须与导出时设定的 input_names 一致 ("inputs")
            return {"inputs": batch.numpy().astype(np.float32)}
        except StopIteration:
            return None


def quantize_fp32_to_int8(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    calibration_dataloader: torch.utils.data.DataLoader,
    quant_format=QuantType.QInt8,
):
    """
    对 FP32 ONNX 模型进行静态 INT8 量化
    """
    print("开始 INT8 静态量化...")
    data_reader = CalibrationDataReader(calibration_dataloader)
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=data_reader,
        quant_format=quant_format,          # QInt8 或 QUInt8
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    print(f"INT8 量化模型已保存至: {int8_onnx_path}")

    # 简单验证
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(int8_onnx_path, providers=['CPUExecutionProvider'])
        print("INT8 量化模型加载成功，推理测试通过。")
    except Exception as e:
        print(f"INT8 量化模型验证失败: {e}")


# ==============================
# 主程序入口 (用户配置区域)
# ==============================
if __name__ == "__main__":
    # -------- INT8量化开关 ----------
    USE_INT8_QUANTIZATION = True   # True: 导出 FP32 后执行 INT8 量化; False: 仅导出 FP32

    # -------- 模型路径与参数 --------
    CHECKPOINT_PATH = "./gearbox-picking/HccePose/obj_01/best_score/0_7769step16500"
    MODELS_INFO_PATH = "./gearbox-picking/models/models_info.json"
    OBJ_ID = 1

    # 导出文件名（可自定义）
    OUTPUT_FP32_ONNX = "./gearbox-picking/HccePose/obj_01/onnx_cache/model_fp32.onnx"
    OUTPUT_INT8_ONNX = "./gearbox-picking/HccePose/obj_01/onnx_cache/model_int8.onnx"

    # 模型结构参数
    EFFICIENTNET_KEY = None       # None 表示使用 ResNet34
    INPUT_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OPSET_VERSION = 17
    DYNAMIC_BATCH = False          # 动态 batch 量化支持有限，建议设为 False

    # -------- 量化校准参数（仅当 USE_INT8_QUANTIZATION=True 时生效）--------
    DATASET_PATH = "./gearbox-picking"
    OBJ_PATH = None               # 自动根据 obj_id 查找
    TRAIN_FOLDER_NAME = "train_pbr"
    PADDING_RATIO = 1.5
    CALIBRATION_RATIO = 0.01      # 使用测试集的 1% 作为校准数据（通常 50~200 张）
    CALIB_BATCH_SIZE = 1
    CALIB_NUM_WORKERS = 0
    LOCAL_RANK = 0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 用户可修改配置结束
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 步骤1: 导出 FP32 ONNX（如果文件已存在则跳过）
    print("=" * 50)
    print("步骤1: 导出 FP32 ONNX 模型")
    print("=" * 50)
    export_hccepose_to_onnx(
        checkpoint_path=CHECKPOINT_PATH,
        output_onnx_path=OUTPUT_FP32_ONNX,
        models_info_path=MODELS_INFO_PATH,
        obj_id=OBJ_ID,
        efficientnet_key=EFFICIENTNET_KEY,
        input_size=INPUT_SIZE,
        device=DEVICE,
        opset_version=OPSET_VERSION,
        dynamic_batch=DYNAMIC_BATCH,
        )

    # 步骤2: 根据开关决定是否量化
    if USE_INT8_QUANTIZATION:
        print("\n" + "=" * 50)
        print("步骤2: 构建校准数据集并进行 INT8 量化")
        print("=" * 50)

        # 构建校准 DataLoader
        calib_loader = get_calibration_dataloader(
            dataset_path=DATASET_PATH,
            obj_id=OBJ_ID,
            obj_path=OBJ_PATH,
            train_folder_name=TRAIN_FOLDER_NAME,
            padding_ratio=PADDING_RATIO,
            ratio=CALIBRATION_RATIO,
            input_size=INPUT_SIZE,
            batch_size=CALIB_BATCH_SIZE,
            num_workers=CALIB_NUM_WORKERS,
            local_rank=LOCAL_RANK,
        )
        print(f"校准数据集大小: {len(calib_loader.dataset)} 张图片")

        # 执行量化
        quantize_fp32_to_int8(
            fp32_onnx_path=OUTPUT_FP32_ONNX,
            int8_onnx_path=OUTPUT_INT8_ONNX,
            calibration_dataloader=calib_loader,
            quant_format=QuantType.QInt8,
        )
    else:
        print("\nINT8 量化已禁用，仅导出 FP32 ONNX 模型。")

    print("\n所有操作完成。")
