@echo off
chcp 65001 > nul
cd /d %~dp0

call .venv\Scripts\activate.bat

REM 渲染数据集（原 s2_p1_gen_pbr_data.sh 逻辑）
set GPU_ID=0
set SCENE_NUM=190
set CC0TEXTURES=E:\python\HCCEPose\cc0textures-512
set DATASET_PATH=.\gearbox-picking
set SCRIPT_PATH=..\s2_p1_gen_pbr_data.py

REM 设置 EGL 设备（与原始脚本一致）
set EGL_DEVICE_ID=%GPU_ID%

REM 进入数据集目录并执行数据生成脚本，过滤包含 "Rendering frame" 的行
pushd %DATASET_PATH%
python %SCRIPT_PATH% --gpu_id %GPU_ID% --cc0textures %CC0TEXTURES% --scene_num %SCENE_NUM% 




pause