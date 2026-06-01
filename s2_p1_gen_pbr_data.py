# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
s2_p1_gen_pbr_data.py 用于生成 PBR 数据，原始脚本改编自 BlenderProc2。
项目链接: https://github.com/DLR-RM/BlenderProc

使用方法（命名参数）:
    cd /path/to/demo-bin-picking
    python /path/to/s2_p1_gen_pbr_data.py --gpu_id 0 --cc0textures /path/to/cc0textures-512 --scene_num 42
'''

import os
import sys
import bpy
import argparse
import blenderproc as bproc
import numpy as np
from tqdm import tqdm
from kasal.utils.io_json import load_json2dict, write_dict2json
import time
import logging
import colorsys
import zipfile
import subprocess


# 配置日志：可通过设置环境变量或修改 level 来关闭 debug 输出
# 关闭方式1：设置环境变量 export BPROC_LOG_LEVEL=INFO
# 关闭方式2：修改下方 level = logging.INFO
logging.basicConfig(
    level=logging.DEBUG,  # 改为 INFO 即可关闭 debug 耗时输出
    format='[%(levelname)s] %(message)s'
)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
DIR_TEXTURES = os.path.join(parent_dir, "cc0textures-512")
ZIP_NAME = "cc0textures-512.zip"
ZIP_PATH = os.path.join(parent_dir, ZIP_NAME)
URL = "https://hf-mirror.com/datasets/SEU-WYL/HccePose/resolve/main/cc0textures-512.zip"

def download_cc0textures():
    if os.path.isdir(DIR_TEXTURES):
        print(f"目录 '{DIR_TEXTURES}' 已存在，跳过下载。")
        return

    print(f"目录 '{DIR_TEXTURES}' 不存在，开始下载...")
    print(f"下载到: {ZIP_PATH}")
    print(f"解压到: {parent_dir}")

    # 跨平台选择 curl 命令
    curl_cmd = "curl.exe" if sys.platform == "win32" else "curl"
    cmd = [curl_cmd, "-L", "-o", ZIP_PATH, "-A", "Mozilla/5.0", URL]
    try:
        subprocess.run(cmd, check=True)
        print("下载完成，开始解压...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zipf:
            zipf.extractall(parent_dir)
        print(f"解压完成，材质已保存到: {DIR_TEXTURES}")
    except subprocess.CalledProcessError as e:
        print(f"curl 下载失败: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}", file=sys.stderr)
        sys.exit(1)


def perturb_vertex_colors_hsv(vcol_data, hue_shift=0.0, sat_shift_range=0.1, val_shift_range=0.2):
    """
    批量修改顶点颜色：在 HSV 空间扰动（默认保持色相不变），只改变深浅和饱和度。
    
    参数:
        vcol_data: bpy.types.MeshLoopColorLayer.data (顶点颜色集合)
        hue_shift: 色相随机偏移范围（±值），默认 0.0 表示完全不变
        sat_shift_range: 饱和度随机偏移范围（±值），默认 0.1
        val_shift_range: 明度随机偏移范围（±值），默认 0.2，控制深浅
    """
    if len(vcol_data) == 0:
        return
    
    # 取第一个顶点的颜色（假设所有顶点颜色相同，效率最高）
    orig_r, orig_g, orig_b, orig_a = vcol_data[0].color
    
    # RGB → HSV
    h, s, v = colorsys.rgb_to_hsv(orig_r, orig_g, orig_b)
    
    # 随机扰动
    new_h = (h + np.random.uniform(-hue_shift, hue_shift)) % 1.0
    new_s = np.clip(s + np.random.uniform(-sat_shift_range, sat_shift_range), 0.0, 1.0)
    new_v = np.clip(v + np.random.uniform(-val_shift_range, val_shift_range), 0.0, 1.0)
    
    # HSV → RGB
    new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    new_color = np.array([new_r, new_g, new_b, orig_a], dtype=np.float32)
    
    # 批量赋值（所有顶点颜色统一为新值）
    flat_colors = np.tile(new_color, (len(vcol_data), 1)).flatten()
    vcol_data.foreach_set("color", flat_colors)
    

if __name__ == '__main__':
    download_cc0textures()
    parser = argparse.ArgumentParser(description='生成 PBR 数据 (BlenderProc)')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU 编号，例如 0')
    parser.add_argument('--cc0textures', type=str, required=True, help='cc0textures 材质库路径')
    parser.add_argument('--scene_num', type=int, default=50, help='生成的场景数量，每个场景渲染 20 帧，总帧数 = scene_num * 20')
    parser.add_argument('--verbose', action='store_true', help='开启详细耗时日志（debug级别）')
    args = parser.parse_args()

    # 如果未指定 --verbose，则关闭 debug 输出
    if not args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    gpu_id = args.gpu_id
    cc_textures_path = args.cc0textures
    num_scenes = args.scene_num

    # 获取当前数据集文件夹路径（运行脚本时需 cd 到数据集目录）
    current_dir = os.path.abspath(os.getcwd())
    dataset_name = os.path.basename(current_dir)
    bop_parent_path = os.path.dirname(current_dir)

    cachedir=os.path.join(os.getcwd(), 'cache')
    os.makedirs(cachedir, exist_ok=True)

    # 加载物体信息
    models_info = load_json2dict(os.path.join(current_dir, 'models', 'models_info.json'))

    if not os.path.exists(os.path.join(current_dir, 'camera.json')):
        write_dict2json(os.path.join(current_dir, 'camera.json'),
                        {
                            "cx": 325.2611083984375,
                            "cy": 242.04899588216654,
                            "depth_scale": 0.1, #在以 uint16 格式存储时：depth_scale = 1 → 分辨率为 1 mm；depth_scale = 0.1 → 分辨率为 0.1 mm
                            "fx": 572.411363389757,
                            "fy": 573.5704328585578,
                            "height": 480,
                            "width": 640
                        })
        

    models_ids = [int(key) for key in models_info.keys()]
    models_ids = np.array(models_ids)

    print('-*' * 10)
    print('bop_parent_path:', bop_parent_path)
    print('dataset_name:', dataset_name)
    print('GPU ID:', gpu_id)
    print('cc0textures path:', cc_textures_path)
    print('Number of scenes:', num_scenes)
    print('-*' * 10)

    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)

    # 初始化 BlenderProc
    bproc.init()
    
    bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)

    # 创建房间平面和光源
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])
    ]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    # 加载材质
    if os.path.basename(cc_textures_path) == 'cc0textures-512':
        print("开始加载cc0textures-512材质")
        cc_textures = bproc.loader.load_512_ccmaterials(cc_textures_path, use_all_materials=True)
    else:
        print("开始加载原版cc0textures材质")
        cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)

    def sample_pose_func(obj: bproc.types.MeshObject):
        min_ = np.random.uniform([-0.15, -0.15, 0.0], [-0.1, -0.1, 0.0])
        max_ = np.random.uniform([0.1, 0.1, 0.4], [0.15, 0.15, 0.6])
        obj.set_location(np.random.uniform(min_, max_))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    bproc.renderer.enable_depth_output(activate_antialiasing=False,output_dir=cachedir)
    bproc.renderer.set_max_amount_of_samples(20)
    bproc.renderer.set_render_devices(desired_gpu_device_type='CUDA', desired_gpu_ids=[gpu_id])
    #相比CUDA，OptiX 在同等 RTX 显卡上通常有 15%-30% 的性能提升
    #bproc.renderer.set_render_devices(desired_gpu_device_type='OPTIX', desired_gpu_ids = [gpu_id]) 

    for i in tqdm(range(num_scenes)):
        t_start = time.time()
        rand_s = np.random.rand()

        # 物体选择逻辑：50% 概率重复选取 10 个物体（允许重复），否则不重复选取最多 30 个
        #if rand_s > 0.5:
        idx_l = np.random.choice(models_ids, size=20, replace=True)
        #else:
        #    idx_l = np.random.choice(models_ids, size=min(len(models_ids), 30), replace=False)
        obj_ids = [int(idx) for idx in idx_l]

        # 步骤 1：加载 BOP 模型
        t0 = time.time()
        target_bop_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=bop_dataset_path,
            mm2m=True,
            obj_ids=obj_ids,
        )
        t1 = time.time()
        logging.debug(f"1. 加载 3D 模型耗时: {t1 - t0:.2f} 秒")


        # 步骤 2：设置材质与初始位姿
        for obj in target_bop_objs:
            obj.set_shading_mode('auto')
            obj.hide(True)
        sampled_target_bop_objs = target_bop_objs
        for obj in sampled_target_bop_objs:
            mat = obj.get_materials()[0]
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))   #粗糙度
            mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))    #高光，用于非金属材质
            #mat.set_principled_shader_value("Metallic", np.random.uniform(0, 1.0))    #金属度，用Metallic时Specular参数通常失效，由金属颜色自动控制高光
            '''
            #增加颜色扰动
            mesh = obj.get_mesh()
            if mesh.vertex_colors:
                vcol_data = mesh.vertex_colors.active.data
                perturb_vertex_colors_hsv(
                    vcol_data,
                    hue_shift=0.0,           # 完全保留原始色相
                    sat_shift_range=0,       # 中性色（黑、白、灰）RGB 三个分量相等，饱和度 S = 0，调整饱和度会导致偏色
                    val_shift_range=0.05      # 明度变化 ±0.25，控制深浅
                    )
            '''
            
            # 性能优化：增加 collision_shape='CONVEX_HULL'
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99, collision_shape='CONVEX_HULL')
            obj.hide(False)

        # 随机光源
        light_plane_material.make_emissive(
            emission_strength=np.random.uniform(3, 6),
            emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
        )
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5, elevation_min=5, elevation_max=89)
        light_point.set_location(location)

        # 随机墙面纹理
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # 随机摆放物体并进行物理模拟
        bproc.object.sample_poses(
            objects_to_sample=sampled_target_bop_objs,
            sample_pose_func=sample_pose_func,
            max_tries=1000
        )
        
        t2 = time.time()
        logging.debug(f"2. 材质和位姿初始化耗时: {t2 - t1:.2f} 秒")

        # 步骤 3：刚体物理仿真
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2,   # 性能优化，等待时间3→2
            max_simulation_time=10,   
            check_object_interval=1, #检查物体运动状态是否趋于稳定的频率
            substeps_per_frame=10,   # 性能优化，20→10 (计算量减半)
            solver_iters=10          # 性能优化，25→10 (计算量减半)
        )
        t3 = time.time()
        logging.debug(f"3. 物理仿真计算(CPU密集)耗时: {t3 - t2:.2f} 秒")

        
        # 步骤 4：建立 BVH 树
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)
        t4 = time.time()
        logging.debug(f"4. BVH树构建(CPU密集)耗时: {t4 - t3:.2f} 秒")


        # 步骤 5：采样相机位姿 (20帧)
        cam_poses = 0
        while cam_poses < 20:
            location = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=0.1,  # 最小物距 0.1 米
                radius_max=0.3,  # 最大物距 0.5 米
                elevation_min=5, # 相机俯仰角 0°相机完全水平
                elevation_max=89
            )
            poi = bproc.object.compute_poi(
                np.random.choice(sampled_target_bop_objs, size=int(round(0.6 * len(obj_ids))), replace=False)
            )
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi - location,
                inplane_rot=np.random.uniform(-3.14159, 3.14159)
            )
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # 距离物体≥0.1米，相机没有被其他物体严重遮挡
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.1}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1
        t5 = time.time()
        logging.debug(f"5. 20帧相机位姿与遮挡计算耗时: {t5 - t4:.2f} 秒")


        # 步骤 6：渲染
        data = bproc.renderer.render(output_dir=cachedir)  #设置缓存路径，否则在windows上会将图片都缓存到C盘根目录下
        t6 = time.time()
        logging.debug(f"6. Cycles与Compositor渲染(GPU密集)耗时: {t6 - t5:.2f} 秒")

        # 步骤 7：保存数据 (磁盘I/O)
        bproc.writer.write_bop(
            bop_parent_path,
            target_objects=sampled_target_bop_objs,
            dataset=dataset_name,
            depth_scale=0.1,
            depths=data["depth"],
            colors=data["colors"],
            color_file_format="JPEG",
            ignore_dist_thres=10
        )
        t7 = time.time()
        logging.debug(f"7. 保存图像与掩码到磁盘(I/O)耗时: {t7 - t6:.2f} 秒")

        
        for obj in sampled_target_bop_objs:
            obj.disable_rigidbody()
            obj.hide(True)
            
        t8 = time.time()
        logging.debug(f"单个场景总耗时: {t8 - t_start:.2f} 秒")
