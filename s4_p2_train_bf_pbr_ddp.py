'''
支持 Kaggle 多卡训练的 HccePose (BF) 训练脚本（命令行版本）。

Kaggle 笔记本中的使用示例：

    单卡（默认）：
    !python s4_p2_train_bf_pbr.py --dataset_path /kaggle/working/HCCEPose/demo-bin-picking

    多卡（例如 2 卡）使用 torchrun（推荐）：
    !torchrun --nproc_per_node=2 s4_p2_train_bf_pbr.py --dataset_path /kaggle/working/HCCEPose/demo-bin-picking --start_obj_id 1 --end_obj_id 1

    使用传统 launch 脚本：
    !python -m torch.distributed.launch --nproc_per_node=2 s4_p2_train_bf_pbr.py --dataset_path /kaggle/working/HCCEPose/demo-bin-picking
'''

import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from HccePose.bop_loader import bop_dataset, train_bop_dataset_back_front, test_bop_dataset_back_front
from HccePose.network_model import HccePose_BF_Net, HccePose_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.metric import add_s
from kasal.bop_toolkit_lib.inout import load_ply


def test(obj_ply, obj_info, net: HccePose_BF_Net, test_loader: torch.utils.data.DataLoader, CUDA_DEVICE):
    net.eval()
    add_list_l = []
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c) in tqdm(enumerate(test_loader)):
        if torch.cuda.is_available():
            rgb_c = rgb_c.to('cuda:' + CUDA_DEVICE, non_blocking=True)
            mask_vis_c = mask_vis_c.to('cuda:' + CUDA_DEVICE, non_blocking=True)
            GT_Front_hcce = GT_Front_hcce.to('cuda:' + CUDA_DEVICE, non_blocking=True)
            GT_Back_hcce = GT_Back_hcce.to('cuda:' + CUDA_DEVICE, non_blocking=True)
            Bbox = Bbox.to('cuda:' + CUDA_DEVICE, non_blocking=True)
            cam_K = cam_K.cpu().numpy()
        with autocast():
            pred_results = net.inference_batch(rgb_c, Bbox)
            pred_mask = pred_results['pred_mask']
            coord_image = pred_results['coord_2d_image']
            pred_front_code_0 = pred_results['pred_front_code_obj']
            pred_back_code_0 = pred_results['pred_back_code_obj']
            pred_front_code = pred_results['pred_front_code']
            pred_back_code = pred_results['pred_back_code']
            pred_front_code_raw = pred_results['pred_front_code_raw'].reshape((-1, 128, 128, 3, 8)).permute((0, 1, 2, 4, 3)).reshape((-1, 128, 128, 24))
            pred_back_code_raw = pred_results['pred_back_code_raw'].reshape((-1, 128, 128, 3, 8)).permute((0, 1, 2, 4, 3)).reshape((-1, 128, 128, 24))
            pred_front_code = torch.cat([pred_front_code, pred_front_code_raw], dim=-1)
            pred_back_code = torch.cat([pred_back_code, pred_back_code_raw], dim=-1)

            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
            pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
            coord_image_np = coord_image.detach().cpu().numpy()
            pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
            for (cam_R_m2c_i, cam_t_m2c_i, pred_m_bf_c_np_i) in zip(cam_R_m2c.detach().cpu().numpy(), cam_t_m2c.detach().cpu().numpy(), pred_m_bf_c_np):
                info_list = solve_PnP_comb(pred_m_bf_c_np_i, train=True)

                for info_id_, info_i in enumerate(info_list):
                    info_list[info_id_]['add'] = add_s(obj_ply, obj_info, [[cam_R_m2c_i, cam_t_m2c_i]], [[info_i['rot'], info_i['tvecs']]])[0]
                add_list = []
                for i_ in range(len(info_list)):
                    info_list_i = itertools.combinations(info_list, len(info_list) - i_)
                    for info_list_i_j in info_list_i:
                        best_add = 0
                        best_s = 0
                        for info_list_i_j_k in info_list_i_j:
                            if info_list_i_j_k['num'] > best_s:
                                best_s = info_list_i_j_k['num']
                                best_add = info_list_i_j_k['add']
                        add_list.append(best_add)
                add_list = np.array(add_list)
                add_list_l.append(add_list)
        torch.cuda.empty_cache()
    add_list_l = np.array(add_list_l)
    add_list_l = np.mean(add_list_l, axis=0)
    print(add_list_l)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    print('max acc id: ', max_acc_id)
    print('max acc: ', max_acc)
    net.train()
    return max_acc_id, max_acc, add_list_l


def parse_args():
    parser = argparse.ArgumentParser(description='HccePose (BF) 训练脚本，支持单卡/多卡分布式训练')

    parser.add_argument('--dataset_path', type=str, required=True, help='数据集根目录路径（例如 ./demo-bin-picking）')
    
    parser.add_argument('--train_folder_name', type=str, default='train_pbr', help='训练数据所在的子文件夹名称，默认为 train_pbr')
    
    # 物体范围
    parser.add_argument('--start_obj_id', type=int, required=True,default=1, help='起始物体 ID（包含）')
    
    parser.add_argument('--end_obj_id', type=int, required=True, default=1, help='结束物体 ID（包含）')
    
    # 训练超参数
    parser.add_argument('--total_iteration', type=int, default=50001, help='总迭代次数，默认 50001')
    
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率，默认 0.0002')
    
    parser.add_argument('--batch_size', type=int, default=24, help='每个 GPU 的 batch size，总 batch = batch_size * GPU数量')
    
    parser.add_argument('--num_workers', type=int, default=12, help='DataLoader 的工作进程数')
    
    parser.add_argument('--log_freq', type=int, default=500, help='保存检查点和测试的间隔迭代次数')
    
    parser.add_argument('--padding_ratio', type=float, default=1.5, help='2D 包围盒缩放比例')
    
    parser.add_argument('--efficientnet_key', type=str, default=None, help='是否使用 EfficientNet，例如 "efficientnet-b3"，默认 None 表示不使用')
    
    parser.add_argument('--test_ratio', type=float, default=0.01, help='从训练集中抽取用于测试的比例，默认 0.01')
    
    # 分布式参数（由 torchrun 或 launch 脚本自动传入）
    parser.add_argument('--local-rank', type=int, default=-1, help='分布式训练时指定本地 GPU 序号，通常不需要手动设置')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ------------------ 分布式环境初始化 ------------------
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        ddp_enabled = True
    elif args.local_rank != -1:
        local_rank = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        ddp_enabled = True
    else:
        ddp_enabled = False
        local_rank = 0   # 单卡情况，使用 GPU 0
        world_size = 1

    CUDA_DEVICE = str(local_rank)
    np.random.seed(local_rank)
    torch.cuda.manual_seed(local_rank)

    is_main_process = (local_rank == 0)

    # 仅在主进程打印配置信息
    if is_main_process:
        print("================== 训练配置 ==================")
        print(f"数据集路径: {args.dataset_path}")
        print(f"训练子文件夹: {args.train_folder_name}")
        print(f"物体范围: {args.start_obj_id} -> {args.end_obj_id}")
        print(f"总迭代次数: {args.total_iteration}")
        print(f"学习率: {args.lr}")
        print(f"单卡 batch size: {args.batch_size}")
        print(f"工作进程数: {args.num_workers}")
        print(f"日志间隔: {args.log_freq}")
        print(f"包围盒缩放: {args.padding_ratio}")
        print(f"EfficientNet key: {args.efficientnet_key}")
        print(f"测试集比例: {args.test_ratio}")
        print(f"分布式模式: {'是' if ddp_enabled else '否'}")
        if ddp_enabled:
            print(f"全局进程数: {world_size}")
        print("=============================================")

    # 加载数据集描述
    bop_dataset_item = bop_dataset(args.dataset_path, local_rank=local_rank)
    train_bop_dataset_back_front_item = train_bop_dataset_back_front(
        bop_dataset_item, args.train_folder_name, padding_ratio=args.padding_ratio,
    )
    # 测试集采样比例从参数传入
    test_bop_dataset_back_front_item = test_bop_dataset_back_front(
        bop_dataset_item, args.train_folder_name, padding_ratio=args.padding_ratio, ratio=args.test_ratio
    )

    for obj_id in range(args.start_obj_id, args.end_obj_id + 1):
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        if is_main_process:
            print(f"开始处理物体 {obj_id}: {obj_path}")
        obj_ply = load_ply(obj_path)
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]

        # 创建保存路径（仅在主进程创建）
        save_path = os.path.join(args.dataset_path, 'HccePose', 'obj_%s' % str(obj_id).rjust(2, '0'))
        best_save_path = os.path.join(save_path, 'best_score')
        if is_main_process:
            os.makedirs(os.path.join(args.dataset_path, 'HccePose'), exist_ok=True)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(best_save_path, exist_ok=True)
            
        # [修复] 等待主进程将目录创建完毕后再让其他进程放行
        if ddp_enabled:
            dist.barrier()

        # 获取 3D 尺寸并移至当前 GPU
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']], dtype=np.float32)).to('cuda:' + CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']], dtype=np.float32)).to('cuda:' + CUDA_DEVICE)

        # [修复核心] 在多卡环境下，强制阻塞非主进程，让主进程先下载预训练权重。
        # 避免多个进程同时下载覆盖相同文件导致的损坏与奔溃 (Exit code 1)。
        if ddp_enabled and local_rank != 0:
            dist.barrier()

        # 定义网络和损失
        loss_net = HccePose_Loss()
        scaler = GradScaler()
        net = HccePose_BF_Net(
            efficientnet_key=args.efficientnet_key,
            input_channels=3,
            min_xyz=min_xyz,
            size_xyz=size_xyz,
        )
        net_test = HccePose_BF_Net(
            efficientnet_key=args.efficientnet_key,
            input_channels=3,
            min_xyz=min_xyz,
            size_xyz=size_xyz,
        )
        
        # [修复核心] 主进程下载完毕并实例化后，释放非主进程直接使用本地缓存
        if ddp_enabled and local_rank == 0:
            dist.barrier()

        net = net.to('cuda:' + CUDA_DEVICE)
        net_test = net_test.to('cuda:' + CUDA_DEVICE)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        # [修复] 尝试加载之前保存的检查点：让"所有进程"统一执行加载以获取相同的 Optimizer 状态！
        # （原代码仅在主进程加载，导致其他卡的动量状态不同步，产生训练发散）
        best_score = 0
        iteration_step = 0
        try:
            checkpoint_info = load_checkpoint(save_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
            best_score = checkpoint_info['best_score']
            iteration_step = checkpoint_info['iteration_step']
            if is_main_process:
                print(f"加载检查点成功：iteration_step={iteration_step}, best_score={best_score}")
        except Exception as e:
            if is_main_process:
                print('未找到检查点，从头开始训练', e)
                
        # 广播起始状态到所有进程 (确保参数在边界情况下的一致性)
        if ddp_enabled:
            best_score_tensor = torch.tensor(best_score, dtype=torch.float32).to('cuda:' + CUDA_DEVICE)
            dist.broadcast(best_score_tensor, src=0)
            best_score = float(best_score_tensor.item())
            
            iteration_step_tensor = torch.tensor(iteration_step, dtype=torch.long).to('cuda:' + CUDA_DEVICE)
            dist.broadcast(iteration_step_tensor, src=0)
            iteration_step = int(iteration_step_tensor.item())

        # 转换为 DDP 模型
        if ddp_enabled:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

        # 更新数据集加载器（每个物体分别构建）
        train_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        if ddp_enabled:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_bop_dataset_back_front_item,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=True,
                drop_last=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_bop_dataset_back_front_item,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                drop_last=True,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_bop_dataset_back_front_item,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True
            )

        test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        test_loader = torch.utils.data.DataLoader(
            test_bop_dataset_back_front_item,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )

        # ------------------ 训练循环 ------------------
        while True:
            end_training = False
            if ddp_enabled:
                train_sampler.set_epoch(iteration_step)
            for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce) in enumerate(train_loader):
                # 测试与保存（仅在主进程执行）
                if is_main_process and (iteration_step % args.log_freq == 0 and iteration_step > 0):
                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        state_dict = net.module.state_dict()
                    else:
                        state_dict = net.state_dict()
                    net_test.load_state_dict(state_dict)
                    max_acc_id, max_acc, add_list_l = test(obj_ply, obj_info, net_test, test_loader, CUDA_DEVICE)
                    if max_acc >= best_score:
                        best_score = max_acc
                        save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, keypoints_=add_list_l)
                    loss_net.print_error_ratio()
                    save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, keypoints_=add_list_l)

                # 数据搬移到当前 GPU
                rgb_c = rgb_c.to('cuda:' + CUDA_DEVICE, non_blocking=True)
                mask_vis_c = mask_vis_c.to('cuda:' + CUDA_DEVICE, non_blocking=True)
                GT_Front_hcce = GT_Front_hcce.to('cuda:' + CUDA_DEVICE, non_blocking=True)
                GT_Back_hcce = GT_Back_hcce.to('cuda:' + CUDA_DEVICE, non_blocking=True)

                with autocast():
                    pred_mask, pred_front_back_code = net(rgb_c)
                    pred_front_code = pred_front_back_code[:, :24, ...]
                    pred_back_code = pred_front_back_code[:, 24:, ...]
                    current_loss = loss_net(pred_front_code, pred_back_code, pred_mask,
                                            GT_Front_hcce, GT_Back_hcce, mask_vis_c)

                    l_l = [
                        3 * torch.sum(current_loss['Front_L1Losses']),
                        3 * torch.sum(current_loss['Back_L1Losses']),
                        current_loss['mask_loss'],
                    ]
                    loss = l_l[0] + l_l[1] + l_l[2]

                # 分布式 NaN 检测
                if ddp_enabled:
                    nan_flag = torch.tensor([int(torch.isnan(loss).any())], device=loss.device)
                    dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
                    if nan_flag.item() > 0:
                        for m in net.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_running_stats()
                        continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                if is_main_process:
                    print(f'dataset:{os.path.basename(args.dataset_path)} - obj{str(obj_id).rjust(2, "0")}',
                          f"iteration_step:{iteration_step}",
                          f"loss_front:{torch.sum(current_loss['Front_L1Losses']).item():.4f}",
                          f"loss_back:{torch.sum(current_loss['Back_L1Losses']).item():.4f}",
                          f"loss_mask:{current_loss['mask_loss'].item():.4f}",
                          f"total_loss:{loss.item():.4f}",
                          flush=True)

                iteration_step += 1
                if iteration_step >= args.total_iteration:
                    end_training = True
                    break
            if end_training:
                if is_main_process:
                    print(f'物体 {obj_id} 训练完成，总迭代次数: {iteration_step}')
                break
        torch.cuda.empty_cache()
