'''
python s3_p2_train_yolo.py --dataset_path /path/to/demo-bin-picking --gpu_num 2 --epochs 50
训练 YOLOv11。训练完成后，文件夹结构如下：
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
|--- train_obj_s
|--- detection
|--- obj_s
|--- yolo11-detection-obj_s.pt
|--- images
|--- labels
|--- yolo_configs
|--- data_objs.yaml
|--- autosplit_train.txt
|--- autosplit_val.txt

'''

import os
import argparse
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 YOLOv11 目标检测模型')
    
    parser.add_argument('--dataset_path', type=str, required=True,help='数据集根目录的路径（例如：/path/to/demo-bin-picking）')
    
    parser.add_argument('--gpu_num', type=int, default=2,help='用于训练的 GPU 数量（默认：2）')
    
    parser.add_argument('--epochs', type=int, default=100,help='训练轮数（默认：100）')
    
    parser.add_argument('--batch_size', type=int, default=None,help='每张 GPU 的批大小。若不设置，自动计算为 12 * gpu_num（默认：None）')
    
    parser.add_argument('--task_suffix', type=str, default='detection',help='任务后缀，用于输出目录和模型命名（默认：detection）')
    
    parser.add_argument('--model_name', type=str, default=None,help='最终模型文件名（默认：yolo11-{task_suffix}-obj_s.pt）')
    
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 使用传入的数据集路径
    dataset_path = args.dataset_path
    gpu_num = args.gpu_num
    epochs = args.epochs

    # 若未指定批大小，则自动计算
    if args.batch_size is None:
        batch_size = 12 * gpu_num
    else:
        batch_size = args.batch_size

    task_suffix = args.task_suffix

    # 基于数据集路径构建相关路径
    dataset_pbr_path = os.path.join(dataset_path, 'train_pbr')
    train_multi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_train', 'train.py')
    data_objs_path = os.path.join(dataset_path, 'yolo11', 'train_obj_s', 'yolo_configs', 'data_objs.yaml')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), task_suffix, "obj_s")
    
    # 确定模型名称
    if args.model_name is None:
        model_name = f"yolo11-{task_suffix}-obj_s.pt"
    else:
        model_name = args.model_name
    
    final_model_path = os.path.join(os.path.dirname(os.path.dirname(data_objs_path)), save_dir, model_name)
    obj_s_path = os.path.dirname(final_model_path)

    # 训练循环：直到输出目录存在才停止
    while True:
        if not os.path.exists(obj_s_path):
            # 构建训练命令
            cmd = (
                f"python {train_multi_path} "
                f"--data_path '{data_objs_path}' "
                f"--epochs {epochs} "
                f"--imgsz 640 "
                f"--batch {batch_size} "
                f"--gpu_num {gpu_num} "
                f"--task {task_suffix}"
            )
            print(f"正在执行命令: {cmd}")
            ret = os.system(cmd)
            if ret != 0:
                print("训练命令执行失败，程序退出。")
                sys.exit(1)
        else:
            print(f"输出目录已存在: {obj_s_path}")
            break
