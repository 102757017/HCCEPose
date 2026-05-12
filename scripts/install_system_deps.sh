#!/bin/bash

%%bash
apt-get update
apt-get install -y wget software-properties-common gnupg2 python3-pip
apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev
apt-get install -y pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl ninja-build
apt-get install -y libsm6 libxrender1 libxext-dev

#jupyter中要使用虚拟显示器
apt-get install -y xvfb

#下载 FreeImage 的动态链接库并安装到 imageio 的插件目录
python -c "import imageio; imageio.plugins.freeimage.download()"