# HCCEPose 项目长期笔记

## 运行环境
- 用户实际跑训练/渲染的 checkout 在 E:\python\HCCEPose（用户名 hewei 的机器）；F:\programming\python\HCCEPose 是本工作区副本，两边手动同步。改完 F: 的代码要提醒同步。

## 已知坑（完整列表见 AGENTS.md「已知坑」）
- pip bpy(3.6.0) import 时前插 5 个 `site-packages/bpy/3.6/scripts/*` 到 sys.path；Windows multiprocessing spawn 原样传给子进程 → 子进程 `import bpy` 解析到 scripts/modules 纯 Python 包 → `ModuleNotFoundError: No module named '_bpy'`。修复：`BopWriterUtility._patch_spawn_sys_path_for_bpy()`（monkeypatch spawn 的 get_preparation_data，只过滤发给子进程的 sys_path）。回归测试：根目录 `test_spawn_bpy_syspath.py`。任何"import bpy + multiprocessing"组合都需同款补丁。
