@echo off
chcp 65001 > nul
cd /d %~dp0

call .venv\Scripts\activate.bat

REM 渲染数据集（原 s2_p1_gen_pbr_data.sh 逻辑）
set GPU_ID=0
set SCENE_NUM=2
set CC0TEXTURES=E:\python\HCCEPose\cc0textures-512
set DATASET_PATH=.\demo-bin-picking
set SCRIPT_PATH=..\s2_p1_gen_pbr_data.py

REM 设置 EGL 设备（与原始脚本一致）
set EGL_DEVICE_ID=%GPU_ID%

REM 进入数据集目录并执行数据生成脚本，过滤包含 "Rendering frame" 的行
pushd %DATASET_PATH%
python %SCRIPT_PATH% --gpu_id %GPU_ID% --cc0textures %CC0TEXTURES% --scene_num %SCENE_NUM% | findstr /v "Rendering frame"
popd

REM 生成 YOLO 数据集
python s3_p1_prepare_yolo_label.py --dataset_path %DATASET_PATH%

REM 训练 YOLO 模型
python s3_p2_train_yolo.py --dataset_path %DATASET_PATH% --gpu_num 1 --batch_size 8 --epochs 1

REM 物体正背面标签制备，windows下需要关闭虚拟显示器
python s4_p1_gen_bf_labels.py --dataset_path %DATASET_PATH%   --no_display

REM 训练 HccePose（分布式）
python -m torch.distributed.launch --nproc_per_node=1 s4_p2_train_bf_pbr_ddp.py --dataset_path %DATASET_PATH% --start_obj_id 1 --end_obj_id 1 --total_iteration 10

pause