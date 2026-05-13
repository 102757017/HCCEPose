'''
python s3_p2_train_yolo.py --dataset_path /path/to/demo-bin-picking --gpu_num 2 --epochs 50

训练 YOLOv11，支持意外中断后自动恢复。
若需重新开始训练，请手动删除最终的 .pt 文件。

训练完成后，文件夹结构如下：
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
            |--- train                  # 训练缓存目录（保留 last.pt 用于恢复）
'''

import os
import argparse
import sys
import time

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 YOLOv11 目标检测模型（支持自动恢复）')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='数据集根目录的路径（例如：/path/to/demo-bin-picking）')
    
    parser.add_argument('--gpu_num', type=int, default=2,
                        help='用于训练的 GPU 数量（默认：2）')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练总轮数（默认：100）')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小，若不设置则自动计算为 12 * gpu_num')
    
    parser.add_argument('--task_suffix', type=str, default='detection',
                        help='任务后缀，用于输出目录和模型命名（默认：detection）')
    
    parser.add_argument('--model_name', type=str, default=None,
                        help='最终模型文件名（默认：yolo11-{task_suffix}-obj_s.pt）')
    
    parser.add_argument('--check_interval', type=int, default=60,
                        help='检测模型文件是否生成的时间间隔（秒），默认60秒')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_path = args.dataset_path
    gpu_num = args.gpu_num
    epochs = args.epochs

    # 自动计算 batch size
    if args.batch_size is None:
        batch_size = 12 * gpu_num
    else:
        batch_size = args.batch_size

    task_suffix = args.task_suffix

    # 构建路径（与 train.py 保存最终模型的路径一致）
    data_objs_path = os.path.join(dataset_path, 'yolo11', 'train_obj_s', 'yolo_configs', 'data_objs.yaml')
    base_dir = os.path.dirname(os.path.dirname(data_objs_path))   # .../yolo11/train_obj_s
    final_model_dir = os.path.join(base_dir, task_suffix, 'obj_s')
    
    if args.model_name is None:
        model_name = f"yolo11-{task_suffix}-obj_s.pt"
    else:
        model_name = args.model_name
    final_model_path = os.path.join(final_model_dir, model_name)

    # 训练脚本路径
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_train', 'train.py')

    # 循环检测最终模型是否生成
    print(f"目标模型文件: {final_model_path}")
    while True:
        if os.path.exists(final_model_path):
            print(f"模型已存在: {final_model_path}，训练完成。")
            break
        else:
            print("模型文件未找到，启动或恢复训练...")
            cmd = (
                f"python {train_script} "
                f"--data_path '{data_objs_path}' "
                f"--epochs {epochs} "
                f"--imgsz 640 "
                f"--batch {batch_size} "
                f"--gpu_num {gpu_num} "
                f"--task {task_suffix}"
            )
            print(f"执行命令: {cmd}")
            ret = os.system(cmd)
            if ret != 0:
                print("训练命令执行失败，程序退出。")
                sys.exit(1)
            # 训练可能因中断而提前退出，但最终模型尚未生成
            # 等待一段时间后再次检查，若仍未生成则继续调用 train.py（自动恢复）
            print(f"训练已结束或中断，等待 {args.check_interval} 秒后重新检查模型文件...")
            time.sleep(args.check_interval)
