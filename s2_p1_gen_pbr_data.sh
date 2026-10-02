#!/usr/bin/env bash
# 渲染 PBR 数据集（BlenderProc）。
#
# 用法:
#   ./s2_p1_gen_pbr_data.sh [选项]
#
# 选项:
#   --gpu_id N         EGL 使用的 GPU 编号（默认 0）
#   --scene_num N      渲染场景数（默认 42）
#   --cc0textures PATH 材质库路径（默认 <仓库根>/cc0textures-512，缺失时自动下载）
#   --dataset_path PATH 数据集根目录（默认 <仓库根>/demo-bin-picking）
#   --script_path PATH s2_p1_gen_pbr_data.py 路径（默认 <仓库根>/s2_p1_gen_pbr_data.py）
#   --no-pause         结束后不等待按键（供其他脚本调用时使用）
#   -h, --help         显示本帮助
#
# 注意: s2_p1_gen_pbr_data.py 直接运行会内存泄漏，必须经由本 wrapper 调用。
# 该 python 脚本把 os.getcwd() 当作数据集根目录，因此下面用子 shell 先 cd 过去。

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Windows venv 的激活脚本在 Scripts/，Linux venv 在 bin/
activate_venv() {
    if [ -n "${_VENV_ACTIVATED:-}" ]; then return 0; fi
    if [ -f "$REPO_ROOT/.venv/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.venv/Scripts/activate"
    elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.venv/bin/activate"
    else
        echo "[warn] 未找到 .venv，沿用当前解释器: $(command -v python)" >&2
    fi
    _VENV_ACTIVATED=1
}

pause_script() {
    [ -t 0 ] || return 0
    read -r -p "按回车键退出..." _
}

# 解析为绝对路径。必须先存进临时变量再判断：若直接写 VAR="$(cd ...)"，
# 命令替换失败时 VAR 已被赋值为空，报错信息就丢了用户传入的原始路径。
# 同理 exit 必须留在主 shell，不能放进 $(...) 的子 shell 里。
abspath() { ( cd "$1" >/dev/null 2>&1 && pwd ); }

usage() {
    cat <<'EOF'
用法: ./s2_p1_gen_pbr_data.sh [选项]

选项:
  --gpu_id N         EGL 使用的 GPU 编号（默认 0）
  --scene_num N      渲染场景数（默认 42）
  --cc0textures PATH 材质库路径（默认 <仓库根>/cc0textures-512，缺失时自动下载）
  --dataset_path PATH 数据集根目录（默认 <仓库根>/demo-bin-picking）
  --script_path PATH s2_p1_gen_pbr_data.py 路径（默认 <仓库根>/s2_p1_gen_pbr_data.py）
  --no-pause         结束后不等待按键（供其他脚本调用时使用）
  -h, --help         显示本帮助
EOF
}

GPU_ID=0
SCENE_NUM=42
cc0textures="$REPO_ROOT/cc0textures-512"
dataset_path="$REPO_ROOT/demo-bin-picking"
script_path="$REPO_ROOT/s2_p1_gen_pbr_data.py"
do_pause=1

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu_id)        GPU_ID="$2"; shift 2 ;;
        --scene_num)     SCENE_NUM="$2"; shift 2 ;;
        --cc0textures)   cc0textures="$2"; shift 2 ;;
        --dataset_path)  dataset_path="$2"; shift 2 ;;
        --script_path)   script_path="$2"; shift 2 ;;
        --no-pause)      do_pause=0; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# 检查顺序：脚本/数据集是廉价检查，放前面；材质库放最后——缺失时可能触发
# 585MB 的大下载，别让前面的路径拼写错误白白浪费这次下载。

# 脚本：只校验目录不够（如 --script_path ../xxx 会把目录指到仓库外、文件不存在），
# 必须确认文件本身存在
_cscript="$(abspath "$(dirname "$script_path")")" || { echo "错误: 脚本目录不存在: $script_path" >&2; exit 1; }
script_path="$_cscript/$(basename "$script_path")"
[ -f "$script_path" ] || { echo "错误: 脚本文件不存在: $script_path" >&2; exit 1; }

_cdataset="$(abspath "$dataset_path")" || { echo "错误: 数据集目录不存在: $dataset_path" >&2; exit 1; }
dataset_path="$_cdataset"

# 路径统一转绝对路径：python 脚本会 chdir，相对路径会失效。
# 材质库缺失时先自动下载（与直接运行 python 脚本的行为对齐），仍失败才报错。
if _cc0textures="$(abspath "$cc0textures")"; then
    cc0textures="$_cc0textures"
else
    echo "[info] 材质库不存在，尝试自动下载: $cc0textures"
    # 绝对路径（/posix 或 C:/ 风格）原样传给下载器；相对路径先锚定到当前 cwd
    case "$cc0textures" in
        /*|[A-Za-z]:[\\/]*) _cc_target="$cc0textures" ;;
        *)                  _cc_target="$(pwd)/$cc0textures" ;;
    esac
    activate_venv
    if ! python "$REPO_ROOT/s2_p0b_download_cc0textures_512.py" "$_cc_target"; then
        echo "错误: 材质库自动下载失败。可手动下载 cc0textures-512.zip 解压到 $_cc_target，" \
             "或用环境变量 CC0TEXTURES_URL 指定镜像后重试" >&2
        exit 1
    fi
    _cc0textures="$(abspath "$cc0textures")" || { echo "错误: 自动下载后仍未找到材质库: $cc0textures" >&2; exit 1; }
    cc0textures="$_cc0textures"
fi

activate_venv

# EGL 设备由 BlenderProc 离屏渲染读取（Windows 走 Mesa EGL，Linux 走 NVIDIA EGL）
export EGL_DEVICE_ID="$GPU_ID"

echo "开始生成 $SCENE_NUM 个场景，每个场景 20 帧，使用 GPU $GPU_ID"
echo "  材质库: $cc0textures"
echo "  数据集: $dataset_path"

# cd 进数据集目录，使脚本内 os.getcwd() 指向数据集根；用子 shell 避免污染当前目录
(
    cd "$dataset_path" || exit 1
    python "$script_path" --gpu_id "$GPU_ID" --cc0textures "$cc0textures" --scene_num "$SCENE_NUM"
)

status=$?
if [ $status -ne 0 ]; then
    echo "渲染失败，退出码 $status" >&2
fi

if [ "$do_pause" -eq 1 ]; then pause_script; fi
exit $status
