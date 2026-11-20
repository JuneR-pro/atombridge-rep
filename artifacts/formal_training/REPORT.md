# 正式训练准备与仓库训练协议

日期：2026-11-18。完成两个正式数据集的归一化统计、本地全参/LoRA 配置和完整训练
batch 检查。本次没有启动 50,000 步训练，没有下载模型或数据。

## 下一步准备结果

| 数据集 | 统计批次数 | 使用的状态帧 | 丢弃尾部帧 | 参与统计的动作向量 |
| --- | --- | --- | --- | --- |
| mani_thermalcycler | 10,919 | 349,408 | 11 | 17,470,400 |
| mani_centrifuge5910 | 17,640 | 564,480 | 25 | 28,224,000 |

统计 batch size 为 32，动作长度为 50，沿用原 `compute_norm_stats.py` 的批次顺序、
不足一批丢弃、输入变换和 `RunningStats`（包括 5000 个直方图区间的分位数估计）。
状态和动作统计均为 32 维，mean/std/q01/q99 全部有限。

为避免无关的视频解码，统计阶段只从 Parquet 读取状态和动作，按 LeRobot 原规则构造
长度 50 的未来动作，轨迹末尾重复最后动作。仍使用原数据变换和原统计计算。
每个场景对照原加载器检查 137 个样本，覆盖所有任务及 episode 边界；变换后的状态、
动作逐值相同，前 3 个完整 batch 的 mean/std/q01/q99 逐值相同。
统计阶段和 batch 检查使用 0 个 loader worker；正式训练配置仍保留默认的 2 个。

统计实际位置：

- `/data1/yanfeng/datasets/atombridge/assets/mani_thermalcycler/norm_stats.json`
- `/data1/yanfeng/datasets/atombridge/assets/mani_centrifuge5910/norm_stats.json`

已为原始与本地的 π0.5 全参/LoRA 配置创建相应 assets 软链接。
两个数据集的完整训练 batch 都可经原训练加载器完成归一化和分词：状态 `[32,32]`、
动作 `[32,50,32]`、token `[32,200]`、各相机 `[32,224,224,3]`，数值全部有限。
两个真实相机为前视和腕部，缺失的第三相机按原适配器补零。

## 策略的任务组织

本地 repo 的默认公开配置按场景各训练一个策略：

- `mani_thermalcycler_pi05`：热循环仪 7 类原子任务联合训练。
- `mani_centrifuge5910_pi05`：离心机 7 类原子任务联合训练。

每个配置读取对应场景完整数据集，没有按原子任务过滤。`prompt_from_task=True`
把每帧的 task_index 映射为语言目标；训练 loader 使用 shuffle。采样单位为帧，
没有按任务等概率重采样，因此轨迹更长的任务会获得更多训练样本。
这两个配置分别训练两个场景策略，没有把两场景的 14 类任务合入同一个配置。

依据：README Fine-tuning；`training/config.py` 第 1144/1157 行；
`training/data_loader.py:create_torch_dataset` 与 `scripts/train.py:main`。

## README 默认 π0.5 全参数微调

| 项目 | 仓库有效值 |
| --- | --- |
| 框架/目标 | JAX + Flax NNX，监督动作 flow matching，均方误差 |
| 模型 | π0.5；PaliGemma/Gemma 2B + Gemma 300M action expert |
| 冻结 | 默认 `nnx.Nothing`，即无冻结参数 |
| 初始化 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| 训练步数 | 50,000 次优化器更新 |
| Global batch size | 32，跨可见设备分配 |
| 输入 | 两路真实 RGB、机器人状态、任务文本；224×224 |
| 动作 | 真实 7 维，先补齐至模型 32 维；每次预测 50 步 |
| 动作表示 | 前 6 维关节目标相对当前状态的增量，第 7 维夹爪保留绝对值 |
| π0.5 状态 | 归一化后离散化到语言 token 输入 |
| 文本长度 | 最大 200 token |
| 归一化 | 1%/99% 分位数，分别对 state/actions 统计 |
| 模型计算 dtype | bfloat16；不代表所有参数和优化器状态都占 bfloat16 空间 |
| 优化器 | AdamW，β1=0.9，β2=0.95，eps=1e-8 |
| Weight decay | 1e-10 |
| 梯度裁剪 | global norm 1.0 |
| 学习率 | warmup 1000 步至 2.5e-5；余弦下降至 2.5e-6 |
| 学习率 decay_steps | 30,000；50,000 步总训练期间，后段保持最低学习率 |
| EMA | 0.99 |
| 随机种子 | 42 |
| DataLoader workers | 2 |
| FSDP | 默认 fsdp_devices=1；多个可见设备下仍可数据并行 |
| 日志/保存 | 每 100 步日志；每 10,000 步及末步保存 |
| checkpoint 保留 | keep_period=5000，默认 max_to_keep=1，周期保护 checkpoint 保留 |
| W&B | 两个场景配置都关闭 |

50,000×32=1,600,000 个训练样本，按当前数据量约为热循环仪 4.58 轮、离心机 2.83 轮，
这里只是样本数折算；代码按 step 控制训练，数据打乱后循环加载，并丢弃每轮不足一批的尾部。
没有额外梯度累积配置。末次 checkpoint 文件夹按从 0 开始的步编号通常为 `49999`。

训练图像增强：非腕部视角随机裁剪到 95% 后恢复尺寸、旋转 ±5°；相机颜色扰动
brightness=0.3、contrast=0.4、saturation=0.5。Flow matching 使用标准高斯噪声，
时间采样 `Beta(1.5,1)*0.999+0.001`，拟合 `noise-actions`。

VLA 默认脚本没有验证集循环、按验证指标选择 best checkpoint 或 early stopping。
当前转换数据全部在 train split。`task_is_complete` 不进入 VLA 的动作损失，而用于
另一套完成检测训练。仓库也有 π0 变体配置，但 README 展示的是 π0.5。

## LoRA 变体

仓库另提供 `mani_thermalcycler_pi05-lora` 和 `mani_centrifuge5910_pi05-lora`。
训练步数、batch size、优化器、学习率、动作表示等与上表一致，区别为：

- 语言 Transformer：attention/FFN 的 LoRA rank=16、alpha=16。
- 动作 Transformer：attention/FFN 的 LoRA rank=32、alpha=32。
- EMA 关闭。
- freeze_filter 冻结 llm 路径下非 LoRA 参数，放行 LoRA 参数；视觉编码器、动作输出投影
  等未被此过滤器覆盖的参数也仍可训练。因此不能描述为“仅 LoRA 参数可训练”。

已用配置的实际过滤器检查代表性参数路径，见 `lora_freeze_check.json`。
之前 3 步 pilot 使用 batch size 1、LoRA；那不是仓库默认全参+batch 32 的正式训练协议。

## 完成检测器的另一套协议

`scripts/intervention_switch/README.md` 将两个场景的数据一起导出，一个共享的
图像/初始图像/文本分类头覆盖两场景任务。冻结 `openai/clip-vit-base-patch32` 骨干，
只训练分类头；hidden_dim=512、dropout=0.2，AdamW lr=1e-3、weight_decay=1e-4，
20 epochs、batch size=32、seed=7。Loss 为带 `negatives/positives` 正类权重的 BCEWithLogits。
默认按完整轨迹划分 train/val/test=70%/15%/15%，加入文本变体与物理成立的矛盾负例，
验证集按 min_precision=0.95 选阈值，再按验证 F1 保存 best.pt 并报告测试指标。
这部分尚未导出或训练；本次只完成 VLA 的正式归一化和配置。

## 本机配置与后续启动

`scripts/formal_training.py` 当前使用已下载并验证的官方初始化：

```
/home/yanfeng/code/EmA/openpi/checkpoints/pi05_base/params
```

本地默认 `fsdp_devices=4`，区别于仓库的默认 1；batch size 32、50,000 步、
优化器、学习率与 EMA 保留原协议。初始化时权重直接按最终参数布局分片，避免全量复制到每卡。

全参和 LoRA 配置快照分别位于各数据集子目录的 `full_config.json`、`lora_config.json`。
`resolved_protocol.json` 记录有效配置、学习率取值、资产和 checkpoint 路径，并已检查
相对仓库配置修改了 FSDP 设备数、实验名与本地存储路径。

准备命令（已完成）：

```bash
JAX_PLATFORMS=cpu bash scripts/atombridge_python.sh scripts/formal_training.py prepare thermal
JAX_PLATFORMS=cpu bash scripts/atombridge_python.sh scripts/formal_training.py prepare centrifuge
```

四卡正式训练入口（默认 GPU 3、5、6、7，启动前检查占用）：

```bash
bash scripts/train_four_gpu.sh thermal --mode full
bash scripts/train_four_gpu.sh centrifuge --mode full
```

可用 `--mode lora` 选择仓库的 LoRA 变体，或显式 `--batch-size`、`--fsdp-devices`
更改资源配置。改动这些参数时需记录为相对默认协议的差异。不要把统计阶段使用 CPU
的 `JAX_PLATFORMS=cpu` 带到正式 GPU 训练命令中。训练前还需按可用 GPU 安排内存和并行；
完整微调的实际短程验证结果见 `artifacts/four_gpu_assessment/TRAINING_READINESS.md`。

默认新实验输出将位于：

- `/data1/yanfeng/checkpoints/atombridge/local_mani_thermalcycler_pi05/formal_50k/`
- `/data1/yanfeng/checkpoints/atombridge/local_mani_centrifuge5910_pi05/formal_50k/`

LoRA 配置名对应追加 `_lora`。checkpoint 包含推理参数、优化器/训练状态和归一化资产，
通过原项目软链接统一写入 `/data1`。脚本对未缓存的远程模型资产直接报错，避免新增下载。
