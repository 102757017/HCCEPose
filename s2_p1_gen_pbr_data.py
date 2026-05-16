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
import bpy
import argparse
import blenderproc as bproc
import numpy as np
from tqdm import tqdm
from kasal.utils.io_json import load_json2dict, write_dict2json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 PBR 数据 (BlenderProc)')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU 编号，例如 0')
    parser.add_argument('--cc0textures', type=str, required=True, help='cc0textures 材质库路径')
    parser.add_argument('--scene_num', type=int, default=50, help='生成的场景数量，每个场景渲染 20 帧，总帧数 = scene_num * 20')
    args = parser.parse_args()

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
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.set_render_devices(desired_gpu_device_type='CUDA', desired_gpu_ids=[gpu_id])

    for i in tqdm(range(num_scenes)):
        rand_s = np.random.rand()

        # 物体选择逻辑：50% 概率重复选取 30 个物体（允许重复），否则不重复选取最多 30 个
        if rand_s > 0.5:
            idx_l = np.random.choice(models_ids, size=30, replace=True)
        else:
            idx_l = np.random.choice(models_ids, size=min(len(models_ids), 30), replace=False)
        obj_ids = [int(idx) for idx in idx_l]

        # 加载物体
        target_bop_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=bop_dataset_path,
            mm2m=True,
            obj_ids=obj_ids,
        )

        # 设置材质和物理属性
        for obj in target_bop_objs:
            obj.set_shading_mode('auto')
            obj.hide(True)
        sampled_target_bop_objs = target_bop_objs
        for obj in sampled_target_bop_objs:
            mat = obj.get_materials()[0]
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
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
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=10,
            check_object_interval=1,
            substeps_per_frame=20,
            solver_iters=25
        )

        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

        # 生成 20 个相机位姿并渲染
        cam_poses = 0
        while cam_poses < 20:
            location = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=0.3,
                radius_max=1.2,
                elevation_min=5,
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
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1

        data = bproc.renderer.render(output_dir=cachedir)  #设置缓存路径，否则在windows上会将图片都缓存到C盘根目录下
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

        for obj in sampled_target_bop_objs:
            obj.disable_rigidbody()
            obj.hide(True)
