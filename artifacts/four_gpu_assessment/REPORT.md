# 4 卡 RTX 4090D 训练评估

更新（2026-11-19）：初始化修复已应用；使用 95% 预分配显存池，真实数据 full/batch32/FSDP4/EMA0.99 已成功执行三步。保存与训练准备结果见 `TRAINING_READINESS.md`。以下为此前的编译评估记录。

官方 pi05_base 已完整下载并成功加载，模型共 3,353,433,872 个参数。本评估使用项目自带 OpenPI 训练代码与配置，global batch size 32，完整微调保留 EMA 0.99。

## 结论

4×24GB 有条件支持完整微调，但不能直接照默认配置启动：需要 `fsdp_devices=4`，并修正初始化时权重在每卡完整复制的问题。LoRA 的显存余量更大。尚未执行完整模型训练步，因此这些结果不构成训练成功或吞吐量实测。

| 情况 | 每卡内存 GiB | 依据 |
| --- | ---: | --- |
| 完整微调，不启用 FSDP | 49.97 | 仅持久训练状态，已超过显存 |
| 完整微调，FSDP 4 | 12.51 | 参数、Adam 状态、EMA 持久占用 |
| 完整微调，FSDP 4，训练步 | 17.40 | GPU 编译器缓冲区峰值估算 |
| 原始初始化，FSDP 4 | 25.00 | GPU 编译器缓冲区峰值估算，超过卡容量 |
| 初始化输入直接分片的原型 | 12.51 | GPU 编译器缓冲区峰值估算 |
| LoRA，FSDP 4，训练步 | 6.22 | GPU 编译器缓冲区峰值估算 |

原始 `init_train_state` 对初始化输入使用 `in_shardings=replicated_sharding`。评估原型将其改成 `(replicated_sharding, state_sharding.params.to_pure_dict())`，避免把全部初始参数复制到每张卡。该改动仅在评估脚本里测试编译，尚未写入实际训练器，也未执行初始化。实际训练前需要应用并运行短程验证。

峰值计算为 argument + output - alias + temp；不包含 CUDA 上下文、通信库额外占用及分配器碎片。编译使用 `--xla_gpu_autotune_level=0`；默认自动调优的额外内存未经验证。不能据此估计 50,000 步耗时。

## 设备与通信

机器实际是 8 张 RTX 4090D，每张 24,564 MiB。评估时只有 GPU 3、5、7 完全空闲，尚无 4 张空闲卡可做完整训练试跑。没有停止其他人的进程。

GPU 0–3 和 4–7 分属两个 NUMA 节点；GPU 间 P2P 读取矩阵报告 CNS（芯片组不支持）。在 GPU 3、5、6、7 上执行了小规模 all-gather 正确性测试，每卡 8 MiB 输入，8 次计时中位数约 4.24 ms，证明该组合能完成集合通信；这不是完整训练吞吐量测试。建议有资源后优先使用同 NUMA 的四卡，例如 4–7。

## 证据

- `full_b32_fsdp4_gpu_trainstep.json`：完整微调训练步编译结果。
- `full_b32_fsdp4_gpu_original_init.json`：原始初始化编译结果。
- `full_b32_fsdp4_gpu_init_shapes_sharded_init.json`：分片初始化原型编译结果。
- `lora_b32_fsdp4_gpu.json`：LoRA 训练步编译结果。
- `gpu_snapshot.csv`、`topology.txt`、`p2p.txt`、`collective_check.json`：硬件与通信记录。
- 评估脚本：`scripts/assess_four_gpu.py`；所有模型训练编译记录的执行训练步数均为 0。

[OpenPI 官方说明](https://github.com/Physical-Intelligence/openpi#requirements)提供单卡微调显存参考，并说明支持 FSDP 多卡训练；本报告数值来自当前项目配置的本地编译评估。
