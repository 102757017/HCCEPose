# 基本用法：指定数据集路径，自动在 models 文件夹下生成 models_info.json
#python s1_p3_obj_infos.py --dataset_path /path/to/demo-bin-picking



import os
import argparse
import numpy as np
from kasal.utils import load_ply_model, load_json2dict, get_all_ply_obj, write_dict2json

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从 PLY 模型文件中提取物体信息并生成 models_info.json')
    parser.add_argument('--dataset_path', type=str, required=True,help='数据集根目录的路径，该目录下必须包含 models 文件夹（例如：/path/to/demo-bin-picking）')
    
    parser.add_argument('--output_path', type=str, default=None,help='生成的 models_info.json 的输出路径。若不指定，则自动保存在 models 文件夹下（默认：dataset_path/models/models_info.json）')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_path = args.dataset_path

    # 获取该文件夹及其所有子文件夹中的所有 PLY 文件
    models_path = get_all_ply_obj(dataset_path)

    # 遍历所有 PLY 文件，计算直径、顶点坐标各轴的最小/最大值、包围盒边长
    # 加载旋转对称先验文件，最后合并所有信息生成 models_info.json
    models_info = {}
    for model_path in models_path:
        ply_info = load_ply_model(model_path)
        model_info = {
            "diameter": float(ply_info['diameter']),
            "max_x": float(np.max(ply_info['vertices'], axis=0)[0]),
            "max_y": float(np.max(ply_info['vertices'], axis=0)[1]),
            "max_z": float(np.max(ply_info['vertices'], axis=0)[2]),
            "min_x": float(np.min(ply_info['vertices'], axis=0)[0]),
            "min_y": float(np.min(ply_info['vertices'], axis=0)[1]),
            "min_z": float(np.min(ply_info['vertices'], axis=0)[2]),
            "size_x": float(np.max(ply_info['vertices'], axis=0)[0] - np.min(ply_info['vertices'], axis=0)[0]),
            "size_y": float(np.max(ply_info['vertices'], axis=0)[1] - np.min(ply_info['vertices'], axis=0)[1]),
            "size_z": float(np.max(ply_info['vertices'], axis=0)[2] - np.min(ply_info['vertices'], axis=0)[2]),
        }
        symmetry_type_dict = None
        sym_type_file = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).split('.')[0] + '_sym_type.json')
        if os.path.exists(sym_type_file):
            symmetry_type_dict = load_json2dict(sym_type_file)
        if symmetry_type_dict is not None:
            if 'symmetries_continuous' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_continuous"] = symmetry_type_dict['current_obj_info']["symmetries_continuous"]
            if 'symmetries_discrete' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_discrete"] = symmetry_type_dict['current_obj_info']["symmetries_discrete"]
        model_id = str(int(os.path.basename(model_path).split('.')[0].split('obj_')[1]))
        models_info[model_id] = model_info

    # 判断该文件夹是否包含物体；若包含，则生成 models_info.json
    if len(models_path) > 0:
        # 确定输出路径
        if args.output_path is not None:
            models_info_path = args.output_path
        else:
            # 默认保存在 models 文件夹下
            models_info_path = os.path.join(os.path.dirname(model_path), 'models_info.json')
        write_dict2json(models_info_path, models_info)
        print(f"已生成 models_info.json 文件：{models_info_path}")
    else:
        print("未找到任何 PLY 模型文件，未生成 models_info.json。")
