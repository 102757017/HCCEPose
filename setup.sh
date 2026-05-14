#!/bin/bash
cd HCCEPose

# Unzip toolkits
unzip -o bop_toolkit.zip
unzip -o blenderproc.zip


pip install uv
df -h
#设置uv缓存在与 /kaggle/working/ 同一个文件系统的路径下，避免触发复制占满working目录
export UV_CACHE_DIR=/home/jupyter/uv-cache
uv sync --extra cuda --group train --no-progress


chmod 777 ./scripts/install_system_deps.sh
./scripts/install_system_deps.sh

#下载 FreeImage 的动态链接库并安装到 imageio 的插件目录
python -c "import imageio; imageio.plugins.freeimage.download()"

#下载材质轻量级替代版本 cc0textures-512
curl -L -o cc0textures-512.zip \
  -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
  "https://hf-mirror.com/datasets/SEU-WYL/HccePose/resolve/main/cc0textures-512.zip"

# 2. 解压到当前目录（会自动创建 cc0textures-512 文件夹）
unzip -q cc0textures-512.zip -d . 
