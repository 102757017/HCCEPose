#!/bin/bash

#渲染数据集
chmod +x s2_p1_gen_pbr_data.sh
./s2_p1_gen_pbr_data.sh \
    --gpu_id 0 \
    --scene_num 2 \
    --cc0textures ../cc0textures-512 \
    --dataset_path ./demo-bin-picking \
    --script_path ../s2_p1_gen_pbr_data.py | grep -v "Rendering frame"


#生成yolo数据集
python s3_p1_prepare_yolo_label.py --dataset_path ./demo-bin-picking


#训练yolo模型
python s3_p2_train_yolo.py --dataset_path ./demo-bin-picking --gpu_num 2 --batch_size 8 --epochs 1

#物体正背面标签制备
python s4_p1_gen_bf_labels.py --dataset_path ./demo-bin-picking


#训练 HccePose
python -m torch.distributed.launch --nproc_per_node=2 s4_p2_train_bf_pbr_ddp.py --dataset_path ./demo-bin-picking --start_obj_id 1 --end_obj_id 1 --total_iteration 501
