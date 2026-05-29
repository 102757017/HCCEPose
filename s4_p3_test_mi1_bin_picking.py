import cv2, os, sys
#sys.path.insert(0, r'E:\python\HCCEPose\bop_toolkit')
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.test_script_utils import print_stage_time_breakdown, save_visual_artifacts
from HccePose.tester import Tester
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 使用绝对路径，避免路径解析错误
    base_dir = r'E:\python\HCCEPose'
    dataset_path = os.path.join(base_dir, 'demo-bin-picking')
    test_img_path = os.path.join(base_dir, 'test_imgs')
    
    bop_dataset_item = bop_dataset(dataset_path)
    
    # 手动修正 dataset_path 属性（如果 bop_dataset 内部存错了）
    bop_dataset_item.dataset_path = dataset_path  # 强制覆盖为正确的路径
    
    obj_id = 1
    CUDA_DEVICE = '0'
    hccepose_vis = True
    save_visualizations = hccepose_vis
    print_stage_timing = False
    hccepose_acceleration = 'pytorch'

    Tester_item = Tester(
        bop_dataset_item,
        hccepose_vis=hccepose_vis,
        CUDA_DEVICE=CUDA_DEVICE,
        hccepose_acceleration=hccepose_acceleration,
    )
    for name in ['000025']:
        file_name = os.path.join(test_img_path, '%s.jpg' % name)
        image = cv2.imread(file_name)
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        results_dict = Tester_item.predict(
            cam_K, image, [obj_id], conf=0.85, confidence_threshold=0.85,
        )
        print_stage_time_breakdown(results_dict, enabled=print_stage_timing, prefix=name)
        save_visual_artifacts([
            (file_name.replace('.jpg', '_show_2d.jpg'), results_dict.get('show_2D_results')),
            (file_name.replace('.jpg', '_show_6d_vis0.jpg'), results_dict.get('show_6D_vis0')),
            (file_name.replace('.jpg', '_show_6d_vis1.jpg'), results_dict.get('show_6D_vis1')),
            (file_name.replace('.jpg', '_show_6d_vis2.jpg'), results_dict.get('show_6D_vis2')),
        ], enabled=save_visualizations)




images = [
    ('2D Results', results_dict.get('show_2D_results')),
    ('6D Vis0', results_dict.get('show_6D_vis0')),
    ('6D Vis1', results_dict.get('show_6D_vis1')),
    ('6D Vis2', results_dict.get('show_6D_vis2'))
]

valid_images = [(name, img) for name, img in images if img is not None]

# 计算需要的行数（比如每行显示2列）
n_cols = 2
n_rows = (len(valid_images) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

for i, (name, img) in enumerate(valid_images):
    axes[i].imshow(img.astype(np.uint8))
    axes[i].set_title(name, fontsize=12)
    axes[i].axis('off')

# 隐藏多余的子图
for j in range(len(valid_images), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
