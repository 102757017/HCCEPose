'''
python s4_p1_gen_bf_labels.py --dataset_path /path/to/data
为 HccePose(BF) 生成训练标签。标签生成完成后，文件夹结构如下：
demo-bin-picking
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front

------------------------------------------------------    

为 HccePose(BF) 生成训练标签。
标签生成完成后，文件夹结构如下：
demo-bin-picking
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
'''

import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='为 HccePose(BF) 生成训练标签。')
    
    parser.add_argument('--dataset_path', type=str,default='./demo-bin-picking',help='数据集的根目录路径。')
    
    parser.add_argument('--bop_toolkit_path', type=str,default='./bop_toolkit',help='bop_toolkit 目录的路径。')
    
    parser.add_argument('--folder_name', type=str, default='train_pbr',help='数据集中要处理的子文件夹名称（例如 train_pbr）。')

    parser.add_argument('--batch_size', type=int, default=16,help='DataLoader 的批大小。')
    
    parser.add_argument('--num_workers', type=int, default=16,help='DataLoader 的工作进程数量。')
    
    parser.add_argument('--obj_ids', type=int, nargs='+', default=None,help='可选：指定要处理的物体 ID 列表。若不指定，则处理所有物体。')
    
    parser.add_argument('--no_display', action='store_true',help='禁用虚拟显示器（当已有真实显示器时使用）。')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 在无头环境中避免“Could not initialize EGL”错误，启动虚拟显示器
    if not args.no_display:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(640, 480))
        display.start()
        print("已启动虚拟显示器。")

    # 将 bop_toolkit 添加到 Python 搜索路径
    if args.bop_toolkit_path not in sys.path:
        sys.path.append(args.bop_toolkit_path)
        print(f"已将 {args.bop_toolkit_path} 添加到 sys.path")

    import torch
    from HccePose.bop_loader import bop_dataset, rendering_bop_dataset_back_front

    # 加载 BOP 数据集
    bop_dataset_item = bop_dataset(args.dataset_path)
    print(f"已加载数据集：{args.dataset_path}")

    # 如果指定了物体 ID，则进行过滤
    if args.obj_ids is not None:
        selected_ids = set(args.obj_ids)
        id_path_pairs = [
            (obj_id, obj_path)
            for obj_id, obj_path in zip(bop_dataset_item.obj_id_list, bop_dataset_item.obj_model_list)
            if obj_id in selected_ids
        ]
        print(f"仅处理以下物体：{args.obj_ids}")
    else:
        id_path_pairs = list(zip(bop_dataset_item.obj_id_list, bop_dataset_item.obj_model_list))
        print(f"处理所有物体：{bop_dataset_item.obj_id_list}")

    # 创建渲染辅助对象
    rendering_item = rendering_bop_dataset_back_front(bop_dataset_item, args.folder_name)

    # 逐个处理每个物体
    for obj_id, obj_path in id_path_pairs:
        print(f"正在处理物体 {obj_id}：{obj_path}")
        rendering_item.update_obj_id(obj_id, obj_path)

        # 用于标签渲染的 DataLoader
        data_gen_loader = torch.utils.data.DataLoader(
            rendering_item,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            worker_init_fn=rendering_item.worker_init_fn
        )

        total_batches = (rendering_item.nSamples + args.batch_size - 1) // args.batch_size
        for batch_idx, (cc_) in enumerate(data_gen_loader):
            if batch_idx % 5 == 0:
                print(f"物体 {obj_id}：批次 {batch_idx}/{total_batches}")
            if batch_idx >= total_batches - 1:
                break

    print("标签生成完成。")

if __name__ == '__main__':
    main()
