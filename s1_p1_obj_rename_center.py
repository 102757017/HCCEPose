#python process_bop.py --input_ply input.ply --obj_id 1 --target_faces 3000

import os
import shutil
import argparse
import pymeshlab as ml
import numpy as np

def modify_ply_texture_filename(input_file, output_file, new_texture_name):
    """
    修改 PLY 文件中的纹理文件名。
    """
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('comment TextureFile'):
                lines[i] = f'comment TextureFile {new_texture_name}\n'
                break
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"修正纹理文件名时出错: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='将 PLY 模型转换为 BOP 格式（减面、中心平移、计算法线、顶点着色）。')
    parser.add_argument('--input_ply', type=str, required=True, help='输入的 PLY 文件路径。')
    parser.add_argument('--obj_id', type=int, required=True, help='物体 ID。')
    parser.add_argument('--output_ply', type=str, default=None, help='输出路径。若不指定，在输入目录下生成 obj_XXXXXX.ply')
    parser.add_argument('--target_faces', type=int, default=10000, help='目标减面数（默认 10000 面）。')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    input_ply = args.input_ply
    obj_id = args.obj_id

    # 保持原有的输出路径逻辑
    if args.output_ply is None:
        output_dir = os.path.dirname(input_ply)
        output_ply = os.path.join(output_dir, f'obj_{obj_id:06d}.ply')
    else:
        output_ply = args.output_ply

    # 1. 加载网格
    ms = ml.MeshSet()
    ms.load_new_mesh(input_ply)
    m = ms.current_mesh()
    print(f"原始面数: {m.face_number()}")

    # 2. 减面 (Simplification)
    # 必须在计算法线前执行，因为减面会重新生成拓扑
    if m.face_number() > args.target_faces:
        print(f"正在进行减面，目标面数: {args.target_faces}...")
        ms.simplification_quadric_edge_collapse_decimation(
            targetfacenum=args.target_faces,
            preserveboundary=True,
            preservenormal=True
        )
        m = ms.current_mesh()

    # 3. 纹理转顶点颜色 (Vertex Coloring)
    # BOP 规范中，顶点着色 (Vertex Color) 兼容性最好
    if m.has_wedge_tex_coord() or m.has_vertex_tex_coord():
        print("将纹理颜色同步至顶点颜色...")
        try:
            ms.compute_color_from_texture_per_vertex()
        except:
            print("警告：顶点颜色转换失败。")

    # 4. 平移模型到原点 (Centering)
    # 重新获取减面后的包围盒中心
    box = m.bounding_box()
    center = box.center()
    print(f"模型中心: {center}，正在移动到原点...")
    ms.compute_matrix_from_translation_rotation_scale(
        translationx=-center[0],
        translationy=-center[1],
        translationz=-center[2]
    )

    # 5. 计算法线 (Normals)
    # 在几何变形和减面完成后计算法线最为准确
    print("重新计算顶点法线...")
    ms.compute_normal_per_vertex()

    # 6. 保存和处理纹理
    m = ms.current_mesh() # 刷新状态
    if m.has_vertex_tex_coord() or m.has_wedge_tex_coord():
        input_texture = input_ply.replace('.ply', '.png')
        output_texture = output_ply.replace('.ply', '.png')
        
        # 复制纹理图片并重命名
        if os.path.exists(input_texture):
            shutil.copy2(input_texture, output_texture)
        
        # 保存带顶点颜色和法线的 PLY
        ms.save_current_mesh(
            output_ply,
            binary=False,
            save_vertex_normal=True,
            save_vertex_color=True,
            save_vertex_coord=True,
            save_wedge_texcoord=False # 已转顶点颜色，可不保存 wedge 坐标
        )
        
        # 修正 PLY 内部对纹理文件的引用
        modify_ply_texture_filename(output_ply, output_ply, os.path.basename(output_texture))
    else:
        # 无纹理模式保存
        ms.save_current_mesh(
            output_ply,
            binary=False,
            save_vertex_normal=True,
            save_vertex_color=True
        )

    print(f"处理完成，输出路径: {output_ply}")
    print(f"最终面数: {m.face_number()}")
