#!/usr/bin/env bash
# 生成 YOLO 标签并训练 YOLOv11 检测器。
#
# 注意: s3_p2_train_yolo.py 会扫描已存在的 yolo11-detection-obj_s.pt 并自动续训。
#       想从零开始训练，必须先删除该文件。
#
# 用法:
#   ./s3_p2_train_yolo.sh [--dataset_path PATH] [--gpu_num N] [--batch_size N] [--epochs N] [--no-pause]
#
# 默认 dataset_path 为 <仓库根>/demo-bin-picking。

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Windows venv 的激活脚本在 Scripts/，Linux venv 在 bin/
activate_venv() {
    if [ -f "$REPO_ROOT/.venv/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.venv/Scripts/activate"
    elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.venv/bin/activate"
    else
        echo "[warn] 未找到 .venv，沿用当前解释器: $(command -v python)" >&2
    fi
}

pause_script() {
    [ -t 0 ] || return 0
    read -r -p "按回车键退出..." _
}

# 见 s2_p1_gen_pbr_data.sh：exit 不能放进 $(...) 子 shell，且要先存临时变量
abspath() { ( cd "$1" >/dev/null 2>&1 && pwd ); }

usage() {
    cat <<'EOF'
用法: ./s3_p2_train_yolo.sh [选项]

选项:
  --dataset_path PATH 数据集根目录（默认 <仓库根>/demo-bin-picking）
  --gpu_num N         YOLO 训练 GPU 数（默认 1）
  --batch_size N      batch size（默认 8）
  --epochs N          训练轮数（默认 1）
  --no-pause          结束后不等待按键
  -h, --help          显示本帮助
EOF
}

dataset_path="$REPO_ROOT/demo-bin-picking"
gpu_num=1
batch_size=8
epochs=1
do_pause=1

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset_path) dataset_path="$2"; shift 2 ;;
        --gpu_num)      gpu_num="$2"; shift 2 ;;
        --batch_size)   batch_size="$2"; shift 2 ;;
        --epochs)       epochs="$2"; shift 2 ;;
        --no-pause)     do_pause=0; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage >&2; exit 1 ;;
    esac
done

_cdataset="$(abspath "$dataset_path")" || { echo "错误: 数据集目录不存在: $dataset_path" >&2; exit 1; }
dataset_path="$_cdataset"

# 依赖 cwd=仓库根（bop_loader.py 会 sys.path.insert(0, os.getcwd())）
cd "$REPO_ROOT" || exit 1
activate_venv

echo "======== 生成 YOLO 数据集 ========"
python s3_p1_prepare_yolo_label.py --dataset_path "$dataset_path" || exit $?

echo
echo "======== 训练 YOLO 模型 ========"
python s3_p2_train_yolo.py --dataset_path "$dataset_path" --gpu_num "$gpu_num" --batch_size "$batch_size" --epochs "$epochs"
status=$?

if [ "$do_pause" -eq 1 ]; then pause_script; fi
exit $status
