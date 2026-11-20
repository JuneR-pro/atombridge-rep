# AtomBridge 复现与 A800 迁移

本仓库基于 [DEFENSE-SEU/Sci-VLA（AtomBridge）](https://github.com/DEFENSE-SEU/Sci-VLA)，保存本地复现过程中增加的工程脚本和训练初始化修复。对应的上游基线提交为 `716e8e3`。

论文：[AtomBridge: Agentic VLA Inference Plugin for Long-Horizon Scientific Manipulation](https://arxiv.org/abs/2602.09430)。原项目安装、数据生成和评估说明保留在 [README.upstream.md](README.upstream.md)。

**当前状态：数据准备完成；未完成正式 50,000 步训练，未宣称复现论文指标。** 当前没有自动训练任务。本文中的训练命令仅供在目标服务器确认环境后手动执行。

## 相对上游修改了什么

| 类别 | 修改 | 原因与影响 |
| --- | --- | --- |
| 初始化实现 | `third_party/openpi/scripts/train.py` 将已加载权重按最终 FSDP 布局直接送入 GPU，并按实际加载的参数筛选分片树 | 避免每卡先保存完整权重造成初始化显存峰值；兼容完整、部分和空权重加载，不改变参数值或模型结构 |
| 并行配置 | `scripts/formal_training.py` 本地默认 `fsdp_devices=4`，支持命令行覆盖 | 适配此前四张 4090；两张 A800 应明确传入 `--fsdp-devices 2` |
| 显存/编译配置 | 四卡启动器预分配 95% 显存，并使用 `--xla_gpu_autotune_level=0` | 为此前 24GB 卡使用的保守配置；目标 A800 的峰值和性能需重新验证 |
| 检查点 I/O | 本地入口将 TensorStore 文件 I/O 并发限制为 2，数据拷贝/压缩并发限制为 4 | 缓解共享 HDD 上的写入等待；保留原检查点格式。使用已验证的 Orbax 0.11.13 内部上下文接口 |
| 本地资产 | 官方 `pi05_base`、tokenizer、归一化统计使用本地路径；数据和输出使用软链接 | 离线训练并将大文件放到独立存储，不将其提交到 Git |
| 数据准备 | 新增渲染完成标记、分块调用原转换器、合并重编号及全量审计 | 提高准备效率和可追溯性；原专家生成器、渲染函数和转换器保持原实现 |
| 归一化 | 统计阶段只读 state/action，复用原变换、批处理和 RunningStats，并检查数值等价性 | 避免统计时解码无关视频；正式训练仍使用原数据加载器 |
| 验证工具 | 增加环境检查、断点下载与校验、显存评估和保存恢复验证 | 辅助复现，不改变训练算法 |

没有修改模型结构、动作损失、优化器更新规则或任务组织。早期 `scripts/thermal_pilot.py` 使用过 RoboCasa 初始化和 LoRA/batch 1；它是历史试验入口，**不是本文正式配置**。

## 训练协议

两个场景分别训练一个策略，每个策略联合训练该场景的 7 类原子任务，通过任务文字区分。

| 项目 | 正式配置 |
| --- | --- |
| 模型 | π0.5，约 33.5 亿参数；官方 `pi05_base` 初始化 |
| 训练方式 | 全参数微调；另保留仓库 LoRA 变体 |
| 步数 | 每个场景 50,000 次更新 |
| Global batch | 32；两卡时各处理 16 个样本 |
| 优化器 | AdamW，β1=0.9、β2=0.95、eps=1e-8、weight decay=1e-10，梯度裁剪 1.0 |
| 学习率 | 1000 步 warmup 到 2.5e-5，余弦下降到 2.5e-6，decay_steps=30000 |
| EMA | 全参 0.99；LoRA 关闭 |
| 输入 | 两路真实 RGB（224×224）、任务文本和机器人状态；缺失视角由原适配器补零 |
| 动作 | 50 步动作段，真实 7 维，补齐到模型 32 维；原 delta action 变换保留 |
| 归一化 | state/action 各自的 q01/q99 分位数 |
| 保存 | 每 10,000 步及末步保存；`keep_period=5000` 保留规则沿用上游 |
| 随机种子 | 42 |

全配置和原协议说明见 [训练协议报告](artifacts/formal_training/REPORT.md) 与 [配置快照](artifacts/formal_training/resolved_protocol.json)。快照中的 `/data1/yanfeng/...` 是原服务器的历史实际路径；启动时配置会根据新服务器的软链接重新解析。

## 数据和权重：手动迁移

这些文件**不在 Git 仓库内**，需要单独复制。训练集必须包含完整的 `data/`、`videos/`、`meta/`。

| 文件 | 原服务器实际路径 | 大小 |
| --- | --- | ---: |
| 热循环仪训练集 | `/data1/yanfeng/datasets/atombridge/lerobot/mani_thermalcycler/` | 0.81 GiB |
| 离心机训练集 | `/data1/yanfeng/datasets/atombridge/lerobot/mani_centrifuge5910/` | 1.78 GiB |
| 两套归一化统计 | `/data1/yanfeng/datasets/atombridge/assets/` | 约 28 KiB |
| 官方初始化权重 | `/home/yanfeng/code/EmA/openpi/checkpoints/pi05_base/` | 11.59 GiB |
| Tokenizer | `/home/yanfeng/code/EmA/AtomBridge/artifacts/thermal_pilot5/openpi_cache/big_vision/paligemma_tokenizer.model` | 小文件 |

两套数据各 700 条轨迹、7 类任务且每类 100 条；热循环仪 349,419 帧，离心机 564,505 帧，共 913,924 帧，50Hz。训练数据与初始化权重合计约 **14.2 GiB**。原始采集数据和分块转换中间文件不属于直接训练的必需资产。原服务器整个 `data` 目录约 9.7 GiB。

建议目标目录布局：

```text
work/
├── atombridge-rep/                  # 本代码仓库
│   ├── data -> /mnt/atombridge/datasets/atombridge
│   ├── checkpoints -> /mnt/atombridge/checkpoints/atombridge
│   └── artifacts/thermal_pilot5/openpi_cache/big_vision/paligemma_tokenizer.model
└── openpi/checkpoints/pi05_base/     # 官方权重，包含 params/ 和 assets/

/mnt/atombridge/datasets/atombridge/
├── lerobot/mani_thermalcycler/
├── lerobot/mani_centrifuge5910/
└── assets/
    ├── mani_thermalcycler/norm_stats.json
    └── mani_centrifuge5910/norm_stats.json
```

复制数据后，在仓库根目录建立软链接。下面 `/mnt/atombridge` 是示例位置，可替换为新服务器实际存储路径：

```bash
mkdir -p /mnt/atombridge/checkpoints/atombridge
ln -s /mnt/atombridge/datasets/atombridge data
ln -s /mnt/atombridge/checkpoints/atombridge checkpoints
ln -s ../../checkpoints third_party/openpi/checkpoints
```

本地训练入口直接使用根目录 `checkpoints`，不依赖 `third_party/openpi/checkpoints`。已有同名目录/链接时应先检查目标，不要盲目覆盖。保持官方权重在仓库上级的 `openpi/checkpoints/pi05_base`，并将 tokenizer 放回上述相对路径。

## Python 环境

需要 Linux、Python 3.11、可用的 NVIDIA 驱动和 CUDA 12 对应 JAX GPU 组件。关键的本地验证版本：

| 包 | 版本 |
| --- | --- |
| jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt | 0.5.3 |
| flax | 0.10.2 |
| orbax-checkpoint | 0.11.13 |
| tensorstore | 0.1.85 |
| torch / torchvision | 2.7.1 / 0.22.1 |
| lerobot | 0.3.3 |
| transformers | 4.53.2 |
| numpy | 1.26.4 |
| mujoco | 3.3.0（由本地覆盖层提供） |

先按 [上游安装说明](README.upstream.md#installation-editable-local-install) 准备基础环境与 FFmpeg；本仓库额外依赖列表为 `requirements-atombridge-overlay.txt`。该列表仅包含覆盖层，**不是完整环境锁文件**。原环境包版本记录见 [active_package_versions.json](artifacts/environment/active_package_versions.json)，其中含其他项目的包，不应整表盲目安装。

安装额外依赖后，将相同包放入项目覆盖层：

```bash
# 在已激活、已安装基础依赖的 Python 3.11 环境中执行。
python -m pip install -r requirements-atombridge-overlay.txt
python -m pip install --target .deps --no-deps -r requirements-atombridge-overlay.txt
export ATOMBRIDGE_BASE_PYTHON="$(command -v python)"
```

必须设置 `ATOMBRIDGE_BASE_PYTHON`：默认值仍是旧服务器解释器的绝对路径。`scripts/atombridge_python.sh` 会自动启用项目覆盖层、vendored OpenPI 源码及离线读取配置。新服务器不应直接复用旧 `.venv` 的绝对路径或旧 GPU 的编译缓存。

只检查环境，不训练：

```bash
JAX_PLATFORMS=cpu bash scripts/atombridge_python.sh scripts/check_atombridge_env.py
```

## 两张 A800 80GB PCIe 的训练入口

两卡应使用 `--fsdp-devices 2`。此前的 `scripts/train_four_gpu.sh` 强制四卡，是 4090 机器的历史入口；**两卡不要使用它**。

在已经完成资产迁移、依赖检查，并准备手动开始训练时，先设置环境：

```bash
export ATOMBRIDGE_BASE_PYTHON="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_autotune_level=0
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

以下沿用已验证的保守显存/编译设置；95% 预分配假定两卡由本任务独占，A800 上的实际峰值及性能需要测量。选择实际可用的 GPU 编号，先用 `nvidia-smi` 检查占用和 `nvidia-smi topo -m` 检查通信拓扑。

分别执行两个场景，使用相同两卡时依次运行：

```bash
bash scripts/atombridge_python.sh -u scripts/formal_training.py train thermal \
  --mode full --fsdp-devices 2 --exp-name a800_2gpu_50k

bash scripts/atombridge_python.sh -u scripts/formal_training.py train centrifuge \
  --mode full --fsdp-devices 2 --exp-name a800_2gpu_50k
```

恢复同一实验时，在完全相同的命令后追加 `--resume`。默认不会覆盖已有实验目录。

输出分别保存在根目录 `checkpoints` 指向的位置：

```text
local_mani_thermalcycler_pi05/a800_2gpu_50k/
local_mani_centrifuge5910_pi05/a800_2gpu_50k/
```

两卡下仅参数、优化器和 EMA 的固定显存约 **25 GiB/卡**，不包括激活、临时缓冲区、通信和 CUDA 开销。建议主机内存 128–256GB，并为两个策略的完整检查点和临时写入预留约 1TB 存储，优先 SSD/NVMe。

没有 A800 实测速率。粗排期可暂按 3–8 秒/步计算，每个策略约 42–111 小时，再计入启动、保存等待和余量；这只是排期假设，不是硬件性能保证。准确估时应在目标机预热后计时 200–500 步，等待 `jax.block_until_ready(...)`，再用 `50000 × 平均每步耗时` 外推，并单独测量完整检查点保存。更大的数据集不自动意味着更长训练：两场景都固定 50,000 步。

## 已通过与尚未通过的验证

- 官方初始化权重 29 个文件，12,441,749,581 字节，全部通过 CRC32C 校验；原生加载后参数形状、类型和有限性检查通过。
- 完整、部分和空权重初始化的四设备小模型回归检查通过，包括参数值和分片布局。
- 四张 4090 上曾成功执行全参、global batch 32、EMA 0.99 的短程更新；不是正式训练。
- 降低 I/O 并发后，完整 π0.5 初始化训练状态在四个 CPU 设备上的保存和恢复通过，包含参数、优化器状态和 EMA；这是 **0 次更新的初始化状态**，不能当作已训练权重使用。
- 限流后的 **GPU 完整检查点保存尚未复测**；两张 A800 上的运行、显存、吞吐和恢复均尚未验证。

详见 [修复与验证记录](artifacts/four_gpu_assessment/TRAINING_READINESS.md)、[数据准备报告](artifacts/formal_dataset/REPORT.md) 和 [官方权重校验](artifacts/pi05_base/REPORT.md)。这些记录是原服务器的历史证据。

## 脚本索引

| 脚本 | 作用 |
| --- | --- |
| `scripts/prepare_formal_dataset.py` | 调用原渲染流程、转换并审计最终数据 |
| `scripts/parallel_convert.py` | 分块调用原转换器并合并 LeRobot 数据 |
| `scripts/audit_raw_collection.py` | 检查原始演示数量、状态和完成标签 |
| `scripts/formal_training.py` | 正式归一化与本地训练配置入口 |
| `scripts/download_pi05_base.py` | 官方权重分块续传及校验 |
| `scripts/verify_pi05_base.py` | 检查已下载权重与模型兼容性 |
| `scripts/verify_fsdp_initialization.py` | 小模型初始化和保存恢复回归检查 |
| `scripts/verify_full_checkpoint_io.py` | CPU 完整初始化训练状态保存恢复验证 |
| `scripts/assess_four_gpu.py` | 四设备静态显存与编译评估 |
| `scripts/thermal_pilot.py` | 历史 RoboCasa/LoRA 小规模试验入口 |

需要重新生成数据、评估长任务或训练完成检测器时，参考原始 README。当前 Git 仓库提交代码和精选小型复现记录；数据、权重、tokenizer、虚拟环境、原始日志和大型中间文件均需独立管理。
