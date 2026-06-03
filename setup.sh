#!/bin/bash
cd HCCEPose

# Unzip toolkits
unzip -o -q bop_toolkit.zip
unzip -o -q blenderproc.zip


pip install uv
#uv sync --extra cpu --extra s1 --extra s2 --extra s3 --extra s4 --extra s5
#uv pip install --system ".[s2]"

chmod 777 ./scripts/install_system_deps.sh
./scripts/install_system_deps.sh

#下载 FreeImage 的动态链接库并安装到 imageio 的插件目录
python -c "import imageio; imageio.plugins.freeimage.download()"


# 下载预训练权重
wget -P ./pre-trained https://hf-mirror.com/datasets/SEU-WYL/HccePose/resolve/main/demo-tex-objs/HccePose/obj_01/best_score/0_8283step50000