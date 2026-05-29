@echo off
chcp 65001 > nul
cd /d %~dp0

call .venv\Scripts\activate.bat


set GPU_ID=0
set SCENE_NUM=2
set CC0TEXTURES=E:\python\HCCEPose\cc0textures-512
set DATASET_PATH=.\demo-bin-picking
set SCRIPT_PATH=..\s2_p1_gen_pbr_data.py


REM 生成 YOLO 数据集
python s3_p1_prepare_yolo_label.py --dataset_path %DATASET_PATH%

REM 训练 YOLO 模型
python s3_p2_train_yolo.py --dataset_path %DATASET_PATH% --gpu_num 1 --batch_size 8 --epochs 1
pause