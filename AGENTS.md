# AGENTS.md

HccePose(BF) —— 6D 位姿估计研究代码库。**以 `README.md` / `README_CN.md` 为准**（1300 行，含完整教程、BOP 权重链接、常见报错排查）。本文件只记录从代码里读出来、README 没写或写错的内容。

## 环境

- Python 3.10（`.python-version` = `3.10`，`requires-python = ">=3.10,<3.13"`）。**不要升到 3.12** —— 固定版本栈（`numpy==1.26.4` / `scipy==1.15.3` / `opencv==4.9.0.80`）只在 3.10 上验证过。
- Windows 用 `C:\Users\hewei\AppData\Local\Programs\Git\git-bash.exe` 跑（`bin\bash.exe` 亦可）。仓库根目录有 `.venv\`，Windows venv 的激活脚本是 `.venv/Scripts/activate`（**没有** `.venv/bin/`，别照抄 Linux 路径）。
- `pyproject.toml` 里有 `[tool.uv] required-environments = ["sys_platform == 'win32'..."]` + `conflicts = [[{extra="cpu"},{extra="cuda"}]]`。依赖按 stage 分组：`uv sync --extra cpu --extra s1 ... --extra s5`。分组是给 CI/分环境安装用的，本地一般直接用现成的 `.venv`。
- **没有 lint / test / typecheck 配置，也没有 CI**。`requirements.txt`（108KB，冻结的完整 pip 输出）不是安装入口，别用它覆盖环境。

## 首次 checkout 必做

`bop_toolkit/` 和 `blenderproc/` 是 zip，**不入 git**（`.gitignore`），必须先解压到仓库根目录：

```bash
unzip bop_toolkit.zip && unzip blenderproc.zip
```

`HccePose/bop_loader.py:24` 直接 `from bop_toolkit.bop_toolkit_lib import ...`，缺了就 `ModuleNotFoundError`。

## 目录与入口

根目录是**扁平的 stage 编号脚本**，不是包：

| 路径 | 内容 |
|---|---|
| `HccePose/` | 核心库。`tester.py`（`Tester` 类：检测+分割+6D 位姿）、`bop_loader.py`（BOP 数据集/标签 dataset）、`network_model.py`、`PnP_solver.py`、`metric.py`、`visualization.py`、`hccepose_acceleration.py`（ONNX/TensorRT） |
| `Refinement/` | RGB-D 精修：`foundationpose.py`、`MegaPose.py`、`foundationpose_acceleration.py` |
| `s1_*` | 物体预处理：PLY 居中改名、KASAL 对称性分析、生成 `models_info.json` |
| `s2_*` | BlenderProc PBR 渲染 |
| `s3_*` | YOLOv11 2D 检测器：标签转换 + 训练 |
| `s4_*` | 前后表面标签生成、位姿训练、BOP 测试、RGB-D/ONNX/TensorRT 推理 |
| `s5_export.py` | ONNX 导出 + INT8 静态量化（配置写在文件底部「用户配置」区，**必须手改**） |
| `yolo_train/` | YOLO 训练辅助（`label.py` / `train.py`），与根目录 `s3_*` 并存 |

数据集目录（`demo-bin-picking/`、`gearbox-picking/` 等）遵循 BOP 规范，权重落在 `demo-bin-picking/HccePose/obj_XX/`。

## 运行约定（最容易踩）

**1. cwd 决定一切 —— 两个脚本相反**

- 绝大多数脚本（`s3_*`、`s4_p1`、`s4_p2`、`s4_p3_*`）**必须从仓库根目录运行**：脚本用 `os.path.dirname(sys.argv[0])` 定位 `demo-bin-picking/`，`bop_loader.py:20` 又 `sys.path.insert(0, os.getcwd())`。
  ```bash
  cd /e/python/HCCEPose && python s4_p3_test_mi10_bin_picking.py
  ```
- **`s2_p1_gen_pbr_data.py` 是例外**：它把 `os.getcwd()` 当作**数据集根目录**（`s2_p1_gen_pbr_data.py:121` `dataset_name = os.path.basename(cwd)`，从 cwd 读 `models/models_info.json`）。必须先 `cd` 进数据集目录，这也是 `s2_p1_gen_pbr_data.sh` / `flow.bat` 里 `cd` / `pushd` 的原因。别"修正"成在根目录跑。

**2. Windows 上要加 `--no_display`**

`s4_p1_gen_bf_labels.py` 用 OpenGL/Egl 离屏渲染，Windows 下弹显示窗口会失败：

```bash
python s4_p1_gen_bf_labels.py --dataset_path ./demo-bin-picking --no_display
```

Linux 下 `bop_loader.py:15-16` 在 import 时自动设 `PYOPENGL_PLATFORM=egl`；Windows 没有这个分支。

**3. 训练用 DDP 专用脚本，不是 `ide_debug` 开关**

README 说的 `ide_debug` 开关在 `s4_p2_train_bf_pbr.py:74`（`True`=单卡，`False`=DDP），但 DDP 实际用独立文件：

```bash
# 单卡
python s4_p2_train_bf_pbr.py --dataset_path ./demo-bin-picking --start_obj_id 1 --end_obj_id 1
# 多卡 DDP（注意是 _ddp 后缀那个文件）
python -m torch.distributed.launch --nproc_per_node=2 s4_p2_train_bf_pbr_ddp.py --dataset_path ./demo-bin-picking --start_obj_id 1 --end_obj_id 1 --total_iteration 501
```

DDP 样本数 = `total_iteration × batch_size × GPU 数`。DDP 不要在 IDE 里直接跑（通信会挂），用 `screen`/`nohup`。

**4. 每个物体一个权重**

HccePose(BF) 训练按 obj 循环，用 `--start_obj_id` / `--end_obj_id` 控制范围；两者相同 = 只训一个物体。`total_iteration` 默认 50000。

**5. YOLO 训练会自动断点续训**

`s3_p2_train_yolo.py` 持续扫描 `yolo11-detection-obj_s.pt`，找到就从该 checkpoint 恢复。**想从头训必须先删掉该文件**。

**6. `Tester` 构造有隐藏副作用**

`HccePose/tester.py:376` 在 `__init__` 里调 `ensure_acceleration_backend_environment`，当 `hccepose_acceleration`/`foundationpose_acceleration` 是 `onnx`/`tensorrt` 时会**自动 `pip install`** onnx / onnxruntime-gpu。所以务必用隔离的 venv/conda 环境。

同时 `tester.py:382` 无条件加载 `<dataset>/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt` —— **推理也必须有 YOLO 权重**，不能只放 HccePose 权重。

## 颜色约定

`cv2.imread` / `VideoCapture` 出来的是 **BGR**，原样传给 `Tester.predict`，不要转。HccePose 内部用 `IMAGENET_MEAN_BGR` / `IMAGENET_STD_BGR`（`bop_loader.py:28-29`，按 BGR 排列的 ImageNet 统计量）。FoundationPose 在 `Refinement_FP.inference_batch` 内部自己 BGR→RGB；MegaPose 喂 RGB 给上游估计器，但 debug 面板是 BGR（给 `cv2.imwrite`）。**不要再加 `COLOR_RGB2BGR` 之类的转换。**

## 已知坑

- **`s4_p3_test_mi1_bin_picking.py` 直接跑会 `NameError`**：matplotlib 绘图代码写在 `if __name__ == '__main__':` 块**之外**（第 56 行起），引用了只在块内定义的 `results_dict`。且第 12 行硬编码 `base_dir = r'E:\python\HCCEPose'`。要用它先修路径并把绘图代码挪进主块。
- **pip bpy 污染 sys.path 会毒死 Windows spawn 子进程（已修，2026-09-30）**：`import bpy` 时 pip wheel 会把 `site-packages/bpy/3.6/scripts/` 下 5 个路径（startup/modules/freestyle/addons）**前插**到 sys.path。Windows multiprocessing 只有 spawn，会把这份 sys.path 原样传给子进程并 runpy 重跑主脚本，此时 `import bpy` 被路径搜索抢先解析到 `scripts/modules/bpy/__init__.py`（纯 Python 包，其 `from _bpy import ...` 在 pyd 引导前执行）→ `ModuleNotFoundError: No module named '_bpy'`。触发点：`bproc.writer.write_bop(num_worker>0)`（BopWriterUtility 的 mask/info Pool，s2 每个场景都会开池）。修复：`BopWriterUtility._patch_spawn_sys_path_for_bpy()` monkeypatch `multiprocessing.spawn.get_preparation_data`，只过滤发给子进程的 sys_path 副本，父进程 sys.path 不动；`_start_mask_worker_pool` 开池前自动调用。回归验证：根目录 `test_spawn_bpy_syspath.py`（默认打补丁应 PASS，`--no-patch` 应复现崩溃）。**注意**：干净解释器里 `import bpy` 的最终归宿本来就是 scripts/modules 的纯 Python 包（pyd 先加载并完成引导后接管 `sys.modules['bpy']`），"bpy.__file__ 指向 .py" 是常态不是 bug；判据是 import 能否成功，且引导完 `_bpy` 不会留在 `sys.modules`。任何"import bpy + multiprocessing"的脚本都踩同款坑，修法可复用。
- **材质库自动下载（cc0textures-512）**：直接跑 `python s2_p1_gen_pbr_data.py`（目标 = `<cwd 父目录>/cc0textures-512`，在 argparse 之前执行）和 `s2_p1_gen_pbr_data.sh` / `flow.sh`（目标 = `--cc0textures` 指定路径，缺失时自动补下而不是报错退出）都会自动下载。下载器是根目录 `s2_p0b_download_cc0textures_512.py`（纯 stdlib 共享模块，不依赖 bpy），特性：先解压到临时目录再整体搬移（中断不留"半个目录"骗过存在性检查）、坏 zip 自动重试一次、环境变量 `CC0TEXTURES_URL` 可覆盖镜像。注意：仓库根那个 1.3MB 的 `cc0textures-512.zip` 是历史下载中断的残骸（完整约 600MB），属坏 zip；重新下载到默认路径时会自动覆盖它。wrapper 的路径参数（`--script_path`/`--cc0textures`/`--dataset_path`）相对**调用时 cwd** 解析（Kaggle notebook 场景注意），脚本文件存在性有前置校验（2026-10-03），坏路径会在触发下载前快速报错。
- **`.bat` 已全部转换为 `.sh`（2026-09-30）**：`flow.sh` / `s2_p1_gen_pbr_data.sh` 是与原 `.bat` **合并后**的跨平台版本（原来 `.sh` 和 `.bat` 各一份、路径硬编码不同，现在统一为单一文件）；`s3_p2_train_yolo.sh` 为新增。三个脚本的共性：
  - `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` 推导路径，**不要写死 `E:\python\HCCEPose`**
  - `activate_venv()` 双分支：Windows venv 在 `.venv/Scripts/activate`（**本仓库实际就是这个**），Linux 在 `.venv/bin/activate`
  - `is_windows()` 靠 `uname -s` 匹配 `MINGW*|MSYS*|CYGWIN*`，用于决定是否给 `s4_p1` 追加 `--no_display`
  - `pause_script()` 用 `[ -t 0 ]` 守卫，非交互终端（如 CI / 被别的脚本调用）自动跳过，所以所有脚本都支持 `--no-pause`
  - **退出码陷阱**：`[ "$x" -eq 1 ] && pause_script` 在条件为假时返回 1，会污染脚本退出码。统一写成 `if ... then ... fi` 再显式 `exit`。
  - 仓库里 `.sh` 在 git index 中是 `100755`（可执行位），且由 `.gitattributes` 的 `*.sh text eol=lf` 强制 LF 换行。**改这些脚本时保持 LF**——CRLF 会让 bash 报 `$'\r': command not found`。
- **`.bat` 已删除（2026-09-30）**：`flow.bat` / `s2_p1_gen_pbr_data.bat` / `s3_p2_train_yolo.bat` 全部移除，只保留 `.sh`。Windows 一律用 git-bash 跑 `.sh`。
- **仓库有 `.gitattributes`**：在本机 `core.autocrlf = true` 的情况下强制 `.sh`/`.py` 为 LF、`.bat`/`.ps1` 为 CRLF，并标记 `.pt`/`.pth`/`.onnx`/`.ply`/`.stl`/`.zip` 为 binary。新增脚本文件时留意它是否被正确归类。
- **`s4_p2_train_bf_pbr_ddp.py` 用的不是仓库里的 `bop_toolkit`**：它 `from kasal.bop_toolkit_lib.inout import load_ply`，走的是 kasal-6d 内置的副本。改 BOP I/O 相关行为时注意有两份。
- **README 引用了不存在的文件**：`requirements-inference.txt`、`hf-dataset-card/README.md`、`pre-trained/`、`2023-10-28-18-33-37/`（FoundationPose 权重目录）在当前 checkout 里都不存在。别照着 README 里的路径去找，需要时用 `scripts/download_hf_assets.py --preset test` 或 `scripts/wget_hf_demo_assets.py` 下载。
- **MegaPose 会在运行时污染环境**：`register_megapose()` 首次调用会 clone `megapose6d` 到 `third_party_megapose6d/`、用 `conda create -p .envs/megapose python=3.9` 建**独立 3.9 环境**、下载模型到 `local_data/megapose-models`，推理通过 `.envs/megapose/bin/python` 子进程跑。**不要**把 MegaPose 的 torch 栈装进主 3.10 环境。首次运行可能几十分钟。
- **FoundationPose 权重不在仓库里**，需自行从 NVlabs 下载放到根目录 `2023-10-28-18-33-37/`（refiner）和 `2024-01-11-20-02-45/`（scorer），各含 `config.yml` + `model_best.pth`。
- **视频推理脚本很慢**：`s4_p3_test_mi10_bin_picking_video.py` / `*_tex_objs_video.py` 遍历整个 `test_videos/`。日常验证用单图脚本。
- `test_imgs_RGBD/` 在 git 里可能只有 `000003_*`（README 提到最小化 clone），需要多帧就下载。
- MegaPose / ONNX / TensorRT / RGB-D 相关脚本首次运行会触发自动下载或自动装包，**别在 IDE 里盲跑**。

## 完整流水线

`flow.sh` 是跨平台（Windows/Linux 通吃）的端到端最小示例，按序执行 5 步：

```bash
./flow.sh                                    # 全用默认值
./flow.sh --dataset_path ./gearbox-picking --scene_num 190 --nproc 2 --gpu_num 2
./flow.sh --skip render --skip yolo_train    # 只重跑部分步骤
```

单步脚本：`s2_p1_gen_pbr_data.sh`（渲染）、`s3_p2_train_yolo.sh`（YOLO 标签+训练）。三个脚本都支持 `--help`。

**顺序不可换**：`s3_p1` 依赖 `s2` 产出的 `train_pbr/`；`s4_p1` 依赖 `s2` + `s1_p3` 的 `models_info.json`；`s4_p2` 依赖 `s4_p1` 产出的 `train_pbr_xyz_GT_{front,back}/`；推理依赖 YOLO 权重 + HccePose 权重。

材质库用 `cc0textures-512`（约 600MB）即可，完整 CC0Textures（44GB）不需要。`s2_p1_gen_pbr_data.py` 直接跑会**内存泄漏**，务必走 shell wrapper（它负责反复调用子进程）。
