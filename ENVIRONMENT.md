# AtomBridge 本机环境

复用 `/home/yanfeng/code/EmA/EmbodiedAgentLab/.venv/bin/python`（Python 3.11），
在项目 `.deps/` 中补齐专用依赖。原虚拟环境未被修改。

## 使用

从当前项目目录执行：

```bash
source scripts/activate_atombridge.sh
python scripts/check_atombridge_env.py
```

或者仅为单条命令启用：

```bash
bash scripts/atombridge_python.sh scripts/check_atombridge_env.py
```

启动脚本优先加载项目 `.deps/`、仓库内 OpenPI 和 openpi-client 源码，并默认设置
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。图像与数据处理的运行缓存放在
`artifacts/runtime/`。它会切换到项目根目录，使仿真插件相对路径可用。

脚本没有选择 GPU。CPU 验证可设置 `JAX_PLATFORMS=cpu`；正式训练时应选择合适的
空闲 GPU，并移除该 CPU 设置。

## 安装版本

新增包固定在 `requirements-atombridge-overlay.txt`。安装时使用了：

```bash
uv pip install \
  --python /home/yanfeng/code/EmA/EmbodiedAgentLab/.venv/bin/python \
  --target .deps --no-deps --link-mode copy \
  -r requirements-atombridge-overlay.txt
```

`--no-deps` 是因为 JAX、PyTorch、LeRobot 等来自已检查的基础环境，不应重复安装。
基础环境版本记录见 `artifacts/offline_audit/base_package_versions.json`，启用后的
版本快照见 `artifacts/environment/active_package_versions.json`。

## 验证范围

2026-11-18 完成以下检查：

- 新增包及 PyTorch、JAX、Flax、LeRobot、CLIP 类均可导入；新增包的直接必需依赖约束通过。
- 本仓库 `mani_thermalcycler_pi05` 训练配置可导入。
- 两个专家模块、MuJoCo 原生插件及热循环仪、离心机场景加载成功。
- TOPPRA 生成有效的双关节轨迹。
- OSMesa 软件渲染成功。
- 生成一条 `open_thermal_cycler_lid` 专家演示，物理成功判定为 True。
- 演示的三个视角渲染成功，转换为一个含 80 帧的 LeRobot episode。
- 转换数据可解码读取；OpenPI 训练数据加载器生成 `[50, 7]` 动作序列，
  经过 VLA 输入适配后得到 `[32]` 状态与 `[50, 32]` 动作。

这条演示为环境验证使用 **5 Hz** 渲染，路径为
`artifacts/environment/smoke_raw/`。转换数据位于
`artifacts/environment/lerobot/atombridge_env_smoke/`。
正式采集应按 README 使用 **50 Hz**，并另建完整数据集。

依赖检查结果保存在 `artifacts/environment/setup_check.json`。

## 后续训练准备

本次只下载软件依赖，没有下载模型、数据，也没有执行训练。软件依赖验证不等同于
模型已经学会原子任务。训练前还需要正式演示数据、归一化统计，以及本地模型资源配置。

VLA 默认训练配置的初始权重指向 `gs://openpi-assets/...`，不受 Hugging Face 离线
开关控制。使用现有权重时，应先把初始化路径改为本地 checkpoint 的 `params`：

```text
/data1/yanfeng/models/robocasa365/pi05_pretrain_human300/multitask_learning/75000/params
```

该 checkpoint 来自 RoboCasa，并非论文的相同初始化。完成检测训练还需整理现有
CLIP 缓存的 tokenizer 和 processor 文件；本次只验证了相关 Python 类的导入。

## 小规模流程验证

随后已完成热循环仪 7 类任务各 5 条演示、50 Hz 渲染与数据转换，并用本地 π0.5
权重完成 3 步 LoRA 训练及 checkpoint 保存。详细结果见
[小规模验证报告](artifacts/thermal_pilot5/REPORT.md)。

## 数据与训练权重存储

数据实际存放在 `/data1/yanfeng/datasets/atombridge`，训练输出实际存放在
`/data1/yanfeng/checkpoints/atombridge`。项目保留以下软链接，旧命令和 pilot 路径仍有效：

| 项目路径 | 实际位置 |
| --- | --- |
| `data` | `/data1/yanfeng/datasets/atombridge` |
| `logs/thermal_cycler_tasks` | `/data1/yanfeng/datasets/atombridge/raw/thermal_cycler_tasks` |
| `logs/centrifuge5910_tasks` | `/data1/yanfeng/datasets/atombridge/raw/centrifuge5910_tasks` |
| `checkpoints` | `/data1/yanfeng/checkpoints/atombridge` |
| `third_party/openpi/checkpoints` | 项目根目录的 `checkpoints` |

激活脚本默认设置 `HF_LEROBOT_HOME` 为项目的 `data/lerobot`，训练与转换使用同一数据根。
README 从 `third_party/openpi` 启动的默认训练输出也会经软链接写入 `/data1`。
显式指定其他 checkpoint 路径的训练命令仍需自行指向这里。

正式数据准备使用 `scripts/prepare_formal_dataset.py` 调用原始渲染函数和转换脚本，
输出为 `mani_thermalcycler`、`mani_centrifuge5910`，进度与检查报告在
`artifacts/formal_dataset/`。渲染可以通过完成标记续跑；转换阶段遇到已存在、
但没有转换完成标记的数据集会停止，避免自动删除数据。

正式的两个数据集现已完成 50 Hz 渲染、分块转换、合并和全量检查，共 1400 条演示、
913,924 帧。分块转换调用原转换脚本，并已通过串行等价性检查。数量、路径和检查结果
见 [正式数据集报告](artifacts/formal_dataset/REPORT.md)。下一步是计算正式数据的
归一化统计并配置训练，本阶段没有启动正式训练。

随后两个正式数据集的归一化与全参/LoRA 本地配置已准备完成，实际统计保存在
`/data1/yanfeng/datasets/atombridge/assets/`。原训练加载器的 batch size 32 输入检查通过。
入口为 `scripts/formal_training.py`；初始化已于 2026-11-19 切换为上级 `openpi` 目录的
官方 `pi05_base`。本地默认 FSDP 4，batch 32、学习率、优化器和 EMA 保留仓库配置。
完整协议、初始化差异和后续命令见 [训练准备与协议报告](artifacts/formal_training/REPORT.md)。

四卡启动入口为 `bash scripts/train_four_gpu.sh thermal --mode full`，另一个场景使用
`centrifuge`。该入口默认 GPU 3、5、6、7，使用 95% 预分配显存池，并在启动前检查占用。
FSDP 初始化已修复为直接分片加载；实际完整微调短程验证和保存记录见
[四卡训练准备](artifacts/four_gpu_assessment/TRAINING_READINESS.md)。本阶段未启动正式 50,000 步训练。
