#!/bin/bash
# 用法: 
#   ./s2_p1_gen_pbr_data.sh --gpu_id 0 --scene_num 42 --cc0textures /path/to/cc0textures --dataset_path /path/to/demo-bin-picking --script_path /path/to/s2_p1_gen_pbr_data.py

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --scene_num)
            SCENE_NUM="$2"
            shift 2
            ;;
        --cc0textures)
            cc0textures="$2"
            shift 2
            ;;
        --dataset_path)
            dataset_path="$2"
            shift 2
            ;;
        --script_path)
            script_path="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$GPU_ID" ] || [ -z "$SCENE_NUM" ] || [ -z "$cc0textures" ] || [ -z "$dataset_path" ] || [ -z "$script_path" ]; then
    echo "错误：缺少必需参数"
    echo "用法: $0 --gpu_id GPU_ID --scene_num SCENE_NUM --cc0textures PATH --dataset_path PATH --script_path PATH"
    exit 1
fi

# 设置 EGL 设备
export EGL_DEVICE_ID=$GPU_ID

# 进入数据集目录，使 python 脚本的 os.getcwd() 指向数据集根目录
cd "$dataset_path"

# 循环生成场景（注意：python 脚本内部的 --scene_num 定义了总场景数，无需在 shell 中循环）
echo "开始生成 $SCENE_NUM 个场景 (每个场景 20 帧)，使用 GPU $GPU_ID"
python "$script_path" --gpu_id "$GPU_ID" --cc0textures "$cc0textures" --scene_num "$SCENE_NUM"