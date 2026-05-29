#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立 ONNX 导出脚本 (基于 HccePose 模型)
从 BOP 格式的 models_info.json 读取物体尺寸。
"""

import json
import os
import torch
from HccePose.network_model import HccePose_BF_Net


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
    efficientnet_key=None,   # 添加此参数，默认 None 表示使用 ResNet34
    input_size: int = 256,
    device: str = "cuda",
    opset_version: int = 17,
    dynamic_batch: bool = True,
):
    # 验证并获取 min_xyz, size_xyz
    if min_xyz is None or size_xyz is None:
        if models_info_path is not None and obj_id is not None:
            min_xyz, size_xyz = load_object_bbox_from_json(models_info_path, obj_id)
        else:
            raise ValueError("必须提供 (min_xyz, size_xyz) 或 (models_info_path, obj_id) 两者之一")

    # 检查 checkpoint 是否存在
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

    # 1. 构建模型并加载权重
    # 注意：efficientnet_key 必须与训练时一致。从警告信息推测训练时使用了 ResNet34 (默认)
    model = HccePose_BF_Net(
        efficientnet_key=efficientnet_key,   # None 表示 ResNet34
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

    # 2. 包装模型 (只导出网络主干)
    net_to_export = model.net if hasattr(model, "net") else model
    export_model = HccePoseExportWrapper(net_to_export).to(device_torch).eval()

    # 3. 准备 dummy 输入
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device_torch)

    # 4. 设置动态轴
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "inputs": {0: "batch_size"},
            "pred_mask_logits": {0: "batch_size"},
            "pred_front_back_code_logits": {0: "batch_size"},
        }

    # 5. 导出 ONNX
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
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
            verbose=False,
        )
    print(f"ONNX 模型已保存至: {output_onnx_path}")

    # 6. 验证
    try:
        import onnx
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型检查通过")
    except ImportError:
        print("未安装 onnx 包，跳过模型验证")



if __name__ == "__main__":
    # ========== 用户可修改区域 ==========
    # 方式一：使用 models_info.json + obj_id
    MODELS_INFO_PATH = "./demo-bin-picking/models/models_info.json"  # 替换为实际路径
    OBJ_ID = 1
    CHECKPOINT_PATH  = "./demo-bin-picking/HccePose/obj_01/best_score/0_9837step824500"
    OUTPUT_ONNX_PATH = "./demo-bin-picking/HccePose/obj_01/onnx_cache/0_9837step824500.onnx"
    EFFICIENTNET_KEY = None   # None 表示使用 ResNet34
    INPUT_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OPSET_VERSION = 17
    DYNAMIC_BATCH = True
    # ===================================
    
    export_hccepose_to_onnx(
        checkpoint_path=CHECKPOINT_PATH,
        output_onnx_path=OUTPUT_ONNX_PATH,
        models_info_path=MODELS_INFO_PATH,
        obj_id=OBJ_ID,
        efficientnet_key=EFFICIENTNET_KEY,
        input_size=INPUT_SIZE,
        device=DEVICE,
        opset_version=OPSET_VERSION,
        dynamic_batch=DYNAMIC_BATCH,
    )
    
    # 也可以直接手动指定 min_xyz, size_xyz 作为备选：
    # min_xyz = torch.tensor([-18.8267, -16.4500, -22.2450], dtype=torch.float32)
    # size_xyz = torch.tensor([37.6542, 32.9000, 44.4900], dtype=torch.float32)
    # export_hccepose_to_onnx(..., min_xyz=min_xyz, size_xyz=size_xyz)
