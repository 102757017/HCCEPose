#1 请提前在MeshLab中顶点着色
#2 请提前在MeshLab中修复法线
#3 请提前将模型定位点移动到原点
#4 导出ply并根据 BOP 规范重命名文件，组织数据集结构




import os
import shutil
import argparse
import pymeshlab as ml
import numpy as np

def modify_ply_texture_filename(input_file, output_file, new_texture_name):
    """
    修改 PLY 文件中的纹理文件名。

    参数:
        input_file: 输入的 PLY 文件路径。
        output_file: 输出的 PLY 文件路径。
        new_texture_name: 新的纹理图片文件名。
    """
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('comment TextureFile'):
                lines[i] = f'comment TextureFile {new_texture_name}\n'
                break
        with open(output_file, 'w') as f:
            f.writelines(lines)
    except FileNotFoundError:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='将 PLY 模型转换为 BOP 格式（中心平移到原点、计算法线、处理纹理）。')
    parser.add_argument('--input_ply', type=str, required=True, help='输入的 PLY 文件路径。')
    parser.add_argument('--obj_id', type=int, required=True, help='物体 ID（若未指定 --output_ply，则用于自动生成输出文件名）。')
    parser.add_argument('--output_ply', type=str, default=None, help='输出的 PLY 文件路径。若不指定，将在输入文件同目录下自动生成为 "obj_XXXXXX.ply"。')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    input_ply = args.input_ply
    obj_id = args.obj_id

    # 若未指定输出路径，则自动生成
    if args.output_ply is None:
        output_dir = os.path.dirname(input_ply)
        output_ply = os.path.join(output_dir, f'obj_{obj_id:06d}.ply')
    else:
        output_ply = args.output_ply

    # 加载网格
    mesh = ml.MeshSet()
    mesh.load_new_mesh(input_ply)

    # 计算顶点法线
    mesh.compute_normal_per_vertex()
    mesh_c = mesh.current_mesh()

    # 通过顶点包围盒计算模型中心
    mesh_vertex_matrix = mesh_c.vertex_matrix().copy()
    vertex_min = np.min(mesh_vertex_matrix, axis=0)
    vertex_max = np.max(mesh_vertex_matrix, axis=0)
    vertex_center = (vertex_min + vertex_max) / 2

    # 平移模型，使中心位于原点
    mesh.compute_matrix_from_translation_rotation_scale(
        translationx=-vertex_center[0],
        translationy=-vertex_center[1],
        translationz=-vertex_center[2],
    )

    # 处理纹理（若存在）
    if mesh_c.texture_number() > 0:
        # 若输入文件同目录下存在纹理图，则复制并重命名
        input_texture = input_ply.replace('.ply', '.png')
        output_texture = output_ply.replace('.ply', '.png')
        if not os.path.exists(input_texture):
            print(f"警告：未找到纹理文件 {input_texture}，跳过复制。")
        else:
            shutil.copy2(input_texture, output_texture)

        # 将 wedge UV 转换为顶点 UV（如果需要）
        if mesh_c.has_wedge_tex_coord():
            mesh.compute_texcoord_transfer_wedge_to_vertex()

        # 保存带纹理的 PLY 文件（不保存 wedge 纹理坐标）
        mesh.save_current_mesh(output_ply,
                               binary=False,
                               save_vertex_normal=True,
                               save_vertex_coord=True,
                               save_wedge_texcoord=False)
        # 修正 PLY 文件中的纹理文件名（MeshLab 无法直接设置此项）
        modify_ply_texture_filename(output_ply, output_ply,
                                    os.path.basename(output_texture))
    else:
        # 保存无纹理的 PLY 文件
        mesh.save_current_mesh(output_ply,
                               binary=False,
                               save_vertex_normal=True)

    print(f"转换后的模型已保存至: {output_ply}")
