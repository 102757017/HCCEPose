# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
The script `train.py` is used to train YOLOv11.  
The original script was adapted from the OpenCV BPC project.  
Project link: https://github.com/opencv/bpc  
We added several augmentation strategies to enhance YOLO's performance.

------------------------------------------------------    

脚本 `train.py` 用于训练 YOLOv11。  
原始脚本改编自 OpenCV BPC 项目。  
项目链接：https://github.com/opencv/bpc  
在此基础上，增加了多种数据增强策略，以提升 YOLO 的性能。
'''

import os
import sys
import argparse
from ultralytics import YOLO
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print(current_dir)

def train_yolo11(task, data_path, gpu_num, epochs, imgsz, batch):
    """
    Train YOLO11 for a specific task ("detection" or "segmentation")
    using Ultralytics YOLO with a single object class.

    Args:
        task (str): "detection" or "segmentation"
        data_path (str): Path to the YOLO .yaml file (e.g. data_obj_11.yaml).
        obj_id (int): The BOP object ID (e.g. 11).
        epochs (int): Number of training epochs.
        imgsz (int): Image size used for training.
        batch (int): Batch size.

    Returns:
        final_model_path (str): Path where the trained model is saved.
    """

    if torch.cuda.is_available():
        device = [i_ for i_ in range(int(gpu_num))]
        print(f"Using GPU(s): {device}")
    else:
        device = 'cpu'
        print("CUDA not available, falling back to CPU.")
    
    if task == "detection":
        pretrained_weights = "yolo26m.pt"
        task_suffix = "detection"
    elif task == "segmentation":
        pretrained_weights = "yolo11n-seg.pt"
        task_suffix = "segmentation"
    else:
        print("Invalid task. Must be 'detection' or 'segmentation'.")
        return None
    if not os.path.exists(data_path):
        print(f"Error: Dataset YAML file not found at {data_path}")
        return None
    print(f"Loading model {pretrained_weights} for {task_suffix} ...")
    
    # 训练输出基础目录：.../yolo11/train_obj_s
    base_dir = os.path.dirname(os.path.dirname(data_path))
    # 固定的训练子目录名（不使用自动递增），便于中断恢复
    train_dir = os.path.join(base_dir, 'train')
    last_pt = os.path.join(train_dir, 'weights', 'last.pt')
    if os.path.exists(last_pt):
        model = YOLO(last_pt)
        resume = True
    else:
        model = YOLO(pretrained_weights)
        resume = False

    model.train(
        data=data_path,                     # 数据集配置文件路径
        epochs=epochs,                      # 训练总轮数
        imgsz=imgsz,                        # 输入图像尺寸
        batch=batch,                        # 批大小
        device=device,                      # 使用的GPU设备列表
        val=True,                           # 是否在验证集上评估
        fraction=1.00,                      # 使用的训练数据比例
        workers=8,                          # 数据加载线程数
        save=True,                          # 是否保存训练检查点
        save_period=1,                      # 每多少轮保存一次模型
        project=base_dir,                   # 训练结果保存的根目录
        name='train',                       # 固定子目录名，避免自动递增
        exist_ok=True,                      # 允许目录已存在（覆盖或恢复）
        close_mosaic=10,                    # 最后10轮关闭马赛克增强
        label_smoothing=0.0,                # 标签平滑因子
        degrees=0.0,                        # 随机旋转角度范围
        translate=0.1,                      # 随机平移比例
        scale=0.50,                         # 随机缩放范围
        shear=0.0,                          # 随机剪切角度
        perspective=0.0,                    # 随机透视变换
        flipud=0.5,                         # 随机上下翻转概率
        fliplr=0.5,                         # 随机左右翻转概率
        mosaic=1.0,                         # 马赛克增强概率
        mixup=1.0,                          # MixUp增强概率
        copy_paste=1.0,                     # 复制粘贴增强概率
        copy_paste_mode='mixup',            # 复制粘贴模式
        resume=resume,                      # 是否从上次训练恢复
        dropout=0.2,                        # Dropout比例
        auto_augment='AugMix',              # 自动增强策略
        freeze=0,                           # 冻结前N层
        multi_scale=0.5,                    # 启用网络输入张量的多尺度分辨率训练
    )

    # 训练完成后，将最终模型保存到固定的任务目录（例如 detection/obj_s/）
    save_dir = os.path.join(base_dir, task_suffix, "obj_s")
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"yolo11-{task_suffix}-obj_s.pt"
    final_model_path = os.path.join(save_dir, model_name)
    model.save(final_model_path)
    print(f"Model saved as: {final_model_path}")
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on a specific dataset and object.")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset YAML file (e.g. idp_codebase/yolo/configs/data_obj_11.yaml).")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation"], default="detection",
                        help="Task type (detection or segmentation).")

    args = parser.parse_args()

    train_yolo11(
        task=args.task,
        data_path=args.data_path,
        gpu_num=args.gpu_num,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )

if __name__ == "__main__":
    main()
