#!/usr/bin/env bash
# HccePose(BF) 端到端最小流水线。
#
# 步骤顺序不可换:
#   1. s2 渲染 PBR 数据集            -> <dataset>/train_pbr/
#   2. s3_p1 生成 YOLO 标签          依赖 train_pbr/
#   3. s3_p2 训练 YOLO 检测器        依赖 yolo11/
#   4. s4_p1 生成正/背面 3D 标签     依赖 train_pbr/ + models/models_info.json
#   5. s4_p2 训练 HccePose(BF)       依赖 train_pbr_xyz_GT_{front,back}/
#
# 用法:
#   ./flow.sh [选项]
#
# 选项:
#   --dataset_path PATH  数据集根目录（默认 <仓库根>/demo-bin-picking）
#   --cc0textures PATH  材质库路径（默认 <仓库根>/cc0textures-512，缺失时自动下载）
#   --scene_num N       渲染场景数（默认 2）
#   --gpu_id N          渲染使用的 GPU（默认 0）
#   --gpu_num N         YOLO 训练 GPU 数（默认 1）
#   --nproc N           HccePose DDP 进程数（默认 1）
#   --start_obj_id N    起始物体 id（默认 1）
#   --end_obj_id N      结束物体 id（默认 1，同值=只训一个物体）
#   --total_iteration N 训练迭代数（默认 501）
#   --skip <step>       跳过某步骤，可重复。step ∈ {render,yolo_label,yolo_train,bf_label,pose_train}
#   --no-pause          结束后不等待按键
#   -h, --help          显示本帮助

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

# git-bash / MSYS 下 s4_p1 的离屏渲染会弹显示窗口，必须加 --no_display
is_windows() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) return 0 ;; *) return 1 ;; esac; }

pause_script() {
    [ -t 0 ] || return 0
    read -r -p "按回车键退出..." _
}

# 见 s2_p1_gen_pbr_data.sh：exit 不能放进 $(...) 子 shell，且要先存临时变量
abspath() { ( cd "$1" >/dev/null 2>&1 && pwd ); }

usage() {
    cat <<'EOF'
用法: ./flow.sh [选项]

选项:
  --dataset_path PATH  数据集根目录（默认 <仓库根>/demo-bin-picking）
  --cc0textures PATH  材质库路径（默认 <仓库根>/cc0textures-512，缺失时自动下载）
  --scene_num N       渲染场景数（默认 2）
  --gpu_id N          渲染使用的 GPU（默认 0）
  --gpu_num N         YOLO 训练 GPU 数（默认 1）
  --nproc N           HccePose DDP 进程数（默认 1）
  --start_obj_id N    起始物体 id（默认 1）
  --end_obj_id N      结束物体 id（默认 1，同值=只训一个物体）
  --total_iteration N 训练迭代数（默认 501）
  --skip <step>       跳过某步骤，可重复。
                      step ∈ {render,yolo_label,yolo_train,bf_label,pose_train}
  --no-pause          结束后不等待按键
  -h, --help          显示本帮助
EOF
}

dataset_path="$REPO_ROOT/demo-bin-picking"
cc0textures="$REPO_ROOT/cc0textures-512"
SCENE_NUM=2
GPU_ID=0
GPU_NUM=1
NPROC=1
start_obj_id=1
end_obj_id=1
total_iteration=501
do_pause=1
SKIP=" render yolo_label yolo_train bf_label pose_train "

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset_path)     dataset_path="$2"; shift 2 ;;
        --cc0textures)      cc0textures="$2"; shift 2 ;;
        --scene_num)        SCENE_NUM="$2"; shift 2 ;;
        --gpu_id)           GPU_ID="$2"; shift 2 ;;
        --gpu_num)          GPU_NUM="$2"; shift 2 ;;
        --nproc)            NPROC="$2"; shift 2 ;;
        --start_obj_id)     start_obj_id="$2"; shift 2 ;;
        --end_obj_id)       end_obj_id="$2"; shift 2 ;;
        --total_iteration)  total_iteration="$2"; shift 2 ;;
        --skip)             SKIP=" $2 $SKIP"; shift 2 ;;
        --no-pause)         do_pause=0; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage >&2; exit 1 ;;
    esac
done

_cdataset="$(abspath "$dataset_path")" || { echo "错误: 数据集目录不存在: $dataset_path" >&2; exit 1; }
dataset_path="$_cdataset"

# 后续 python 脚本依赖 cwd=仓库根（bop_loader.py 会 sys.path.insert(0, os.getcwd())）
cd "$REPO_ROOT" || exit 1
activate_venv

step() { echo; echo "======== $* ========"; }
skipped() { case "$SKIP" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if skipped render; then echo "跳过: 渲染 PBR 数据集"; else
    step "1/5 渲染 PBR 数据集"
    # grep 过滤 BlenderProc 每帧刷屏的 "Rendering frame" 日志
    "$REPO_ROOT/s2_p1_gen_pbr_data.sh" \
        --gpu_id "$GPU_ID" \
        --scene_num "$SCENE_NUM" \
        --cc0textures "$cc0textures" \
        --dataset_path "$dataset_path" \
        --script_path "$REPO_ROOT/s2_p1_gen_pbr_data.py" \
        --no-pause | grep -v "Rendering frame"
fi

if skipped yolo_label; then echo "跳过: 生成 YOLO 标签"; else
    step "2/5 生成 YOLO 数据集"
    python s3_p1_prepare_yolo_label.py --dataset_path "$dataset_path"
fi

if skipped yolo_train; then echo "跳过: 训练 YOLO 检测器"; else
    step "3/5 训练 YOLO 模型"
    python s3_p2_train_yolo.py --dataset_path "$dataset_path" --gpu_num "$GPU_NUM" --batch_size 8 --epochs 1
fi

if skipped bf_label; then echo "跳过: 生成正/背面 3D 标签"; else
    step "4/5 物体正背面标签制备"
    bf_args=(--dataset_path "$dataset_path")
    if is_windows; then
        echo "[info] 检测到 Windows 环境，追加 --no_display 关闭虚拟显示器"
        bf_args+=(--no_display)
    fi
    python s4_p1_gen_bf_labels.py "${bf_args[@]}"
fi

if skipped pose_train; then echo "跳过: 训练 HccePose(BF)"; else
    step "5/5 训练 HccePose（分布式）"
    python -m torch.distributed.launch --nproc_per_node="$NPROC" s4_p2_train_bf_pbr_ddp.py \
        --dataset_path "$dataset_path" \
        --start_obj_id "$start_obj_id" \
        --end_obj_id "$end_obj_id" \
        --total_iteration "$total_iteration"
fi

step "流水线结束"
if [ "$do_pause" -eq 1 ]; then pause_script; fi
exit 0
