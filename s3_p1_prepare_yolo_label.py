# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
The script `s3_p1_prepare_yolo_label.py` is used to convert a BOP-format PBR training dataset into the YOLO format.  
After specifying the path and running this script, a new folder named `yolo11` will be created under the dataset directory.  
The following is an example of the resulting folder structure:
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
|--- train_obj_s
|--- images
|--- labels
|--- yolo_configs
|--- data_objs.yaml
|--- autosplit_train.txt
|--- autosplit_val.txt

text

------------------------------------------------------    

脚本 `s3_p1_prepare_yolo_label.py` 的功能是将 BOP 格式的 PBR 训练数据集转换为 YOLO 格式的数据集。  
运行后会在数据集目录下生成一个名为 `yolo11` 的文件夹。  
以下示例展示了生成后的文件夹结构：
demo-bin-picking
|--- models
|--- train_pbr
|--- yolo11
|--- train_obj_s
|--- images
|--- labels
|--- yolo_configs
|--- data_objs.yaml
|--- autosplit_train.txt
|--- autosplit_val.txt

text
'''

import os
import json
import argparse
from yolo_train.label import convert_train_pbr_2_yolo, generate_yaml

def parse_args():
    parser = argparse.ArgumentParser(description='将 BOP 格式的 PBR 训练数据集转换为 YOLO 格式')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集根目录路径（例如 /path/to/demo-bin-picking）')
    parser.add_argument('--dataset_name', type=str, default=None, help='数据集名称，默认从 dataset_path 提取')
    parser.add_argument('--output_folder', type=str, default='yolo11', help='YOLO 格式输出文件夹名称，默认为 yolo11')
    parser.add_argument('--train_subfolder', type=str, default='train_obj_s', help='训练数据子文件夹名称，默认为 train_obj_s')
    parser.add_argument('--pbr_folder', type=str, default='train_pbr', help='PBR 数据文件夹名称，默认为 train_pbr')
    parser.add_argument('--models_folder', type=str, default='models', help='模型文件夹名称，默认为 models')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 获取数据集名称（如果未指定，则使用 dataset_path 的 basename）
    if args.dataset_name is None:
        dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    else:
        dataset_name = args.dataset_name
    
    # 构建路径
    pbr_path = os.path.join(args.dataset_path, args.pbr_folder)
    models_path = os.path.join(args.dataset_path, args.models_folder)
    output_path = os.path.join(args.dataset_path, args.output_folder, args.train_subfolder)
    
    print(f"[INFO] 数据集路径: {args.dataset_path}")
    print(f"[INFO] 数据集名称: {dataset_name}")
    print(f"[INFO] PBR 数据路径: {pbr_path}")
    print(f"[INFO] 模型路径: {models_path}")
    print(f"[INFO] YOLO 输出路径: {output_path}")
    
    # 从 models_info.json 文件提取所有物体 ID
    obj_id_list = []
    models_info_path = os.path.join(models_path, 'models_info.json')
    if not os.path.exists(models_info_path):
        raise FileNotFoundError(f"未找到 models_info.json 文件: {models_info_path}")
    
    with open(models_info_path, "r") as f:
        scene_gt_data = json.load(f)
    for key_ in scene_gt_data:
        obj_id_list.append(key_)
    
    print(f"[INFO] 找到物体 ID: {obj_id_list}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 将 BOP 格式的 train_pbr 数据集转换为 YOLO 格式的数据集
    convert_train_pbr_2_yolo(pbr_path, output_path, obj_id_list)
    generate_yaml(output_path, obj_id_list)
    print("[INFO] 数据集准备完成！")
