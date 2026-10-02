# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# 2026-10-03: 从 s2_p1_gen_pbr_data.py 的内联下载逻辑抽出为独立模块，供 python
# 脚本与 .sh wrapper 共用（wrapper 在材质库缺失时自动调用本模块）。

'''
下载并解压 cc0textures-512 材质库（纯标准库实现，不依赖 bpy/blenderproc）。

用法:
    python s2_p0b_download_cc0textures_512.py <目标目录>

环境变量 CC0TEXTURES_URL 可覆盖下载源（默认 hf-mirror 上的 cc0textures-512.zip）。
'''

import os
import shutil
import subprocess
import sys
import zipfile

DEFAULT_URL = "https://hf-mirror.com/datasets/SEU-WYL/HccePose/resolve/main/cc0textures-512.zip"


def _curl_cmd():
    # Windows 10+ 自带 curl.exe，Linux/macOS 用系统 curl
    return "curl.exe" if sys.platform == "win32" else "curl"


def download_cc0textures(dest_dir, url=None):
    """下载并解压 cc0textures-512 到 dest_dir；已存在则跳过。返回 dest_dir。

    与旧版内联实现（直接 extractall 到父目录）的差异:
      - zip 下载成功后先解压到同目录的临时目录，再整体搬移到 dest_dir，
        下载/解压中断不会留下"半个目录"骗过下次的存在性检查；
      - zip 损坏（下载中断的典型产物）自动删除并重试一次。
    """
    dest_dir = os.path.abspath(dest_dir)
    if os.path.isdir(dest_dir):
        print(f"目录 '{dest_dir}' 已存在，跳过下载。")
        return dest_dir

    url = url or os.environ.get("CC0TEXTURES_URL", DEFAULT_URL)
    parent = os.path.dirname(dest_dir) or os.curdir
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, os.path.basename(dest_dir) + ".zip")

    tmp_dir = zip_path + ".extracting"
    last_err = None
    for attempt in (1, 2):
        cmd = [_curl_cmd(), "-L", "-o", zip_path, "-A", "Mozilla/5.0", url]
        print(f"下载到: {zip_path} (尝试 {attempt}/2)")
        subprocess.run(cmd, check=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            last_err = None
            break
        except zipfile.BadZipFile as err:
            last_err = err
            print(f"[warn] zip 损坏（多为下载中断），删除后重试: {err}", file=sys.stderr)
            try:
                os.remove(zip_path)
            except OSError:
                pass
    if last_err is not None:
        raise RuntimeError(f"cc0textures-512 下载/解压连续两次失败: {last_err}")

    # zip 内部可能自带顶层目录（cc0textures-512/...），也可能是散文件；统一对齐 dest_dir
    entries = os.listdir(tmp_dir)
    src = tmp_dir
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
        src = os.path.join(tmp_dir, entries[0])
    shutil.move(src, dest_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"解压完成，材质已保存到: {dest_dir}")
    return dest_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python s2_p0b_download_cc0textures_512.py <目标目录>", file=sys.stderr)
        sys.exit(2)
    download_cc0textures(sys.argv[1])
