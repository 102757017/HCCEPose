#!/bin/bash
git clone https://github.com/102757017/HCCEPose.git
cd HCCEPose

# Unzip toolkits
unzip bop_toolkit.zip
unzip blenderproc.zip

chmod 777 ./HCCEPose/scripts/install_system_deps.sh
./HCCEPose/scripts/install_system_deps.sh

pip install uv
cd HCCEPose
uv pip install -r pyproject.toml --no-progress --system


#下载材质轻量级替代版本 cc0textures-512
curl -L -o cc0textures-512.zip \
  -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
  "https://huggingface.co/datasets/SEU-WYL/HccePose/resolve/main/cc0textures-512.zip"

# 2. 解压到当前目录（会自动创建 cc0textures-512 文件夹）
unzip -q cc0textures-512.zip -d . 