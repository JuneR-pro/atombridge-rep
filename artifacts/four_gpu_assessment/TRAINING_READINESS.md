# 四卡训练修复与验证（2026-11-19）

当前仅修复和验证；遵照用户最新要求，不再启动训练。此前短程训练已结束，没有运行正式 50,000 步实验。

## 已实施修改

- `third_party/openpi/scripts/train.py`：初始化输入直接使用最终参数的 FSDP 分片，按实际已加载参数筛选分片树，兼容完整权重、缺少 LoRA 参数的部分权重和空加载器。
- `scripts/formal_training.py`：初始化切换为上级 `openpi/checkpoints/pi05_base/params`，本地默认 FSDP 4；新增 `--num-train-steps` 供独立短程验证使用。
- `scripts/train_four_gpu.sh`：默认使用 GPU 3、5、6、7；JAX 预分配 95% 显存，关闭 GPU 自动调优；启动前检查四卡占用，超过 512 MiB 就拒绝启动。可用 `CUDA_VISIBLE_DEVICES` 指定其他四卡。
- `scripts/formal_training.py` 将 TensorStore 的文件 I/O 并发从默认 128 限制为 2，数据拷贝/压缩并发为 4，避免共享 HDD 请求队列被大量检查点写入线程压满。该局部配置使用现有 Orbax 0.11.13 的 TensorStore 上下文接口，不修改系统磁盘设置。
- 两场景 full/LoRA 配置快照与有效协议已更新。Global batch 32、50,000 步、AdamW、学习率、EMA 等保持各自原配置，资源配置改为 FSDP 4。

## 验证记录

`scripts/verify_fsdp_initialization.py` 在四个虚拟 CPU 设备上通过完整、部分及空权重三种加载测试，检查参数数值、EMA 和实际分片布局。

第一轮真实数据试跑通过了初始化，但默认非预分配显存池在训练临时缓冲区申请约 6.26 GiB 时 OOM。改为 95% 预分配后，真实四卡完整微调成功执行三步，global batch 32、EMA 0.99，loss 为 0.8160 / 0.7838 / 0.7291，梯度范数有限。日志：`artifacts/formal_training/fsdp4_pool_smoke_20261119.log`。另一轮带堆栈诊断的三步试跑也成功执行更新。

这两轮 GPU 保存未完成：诊断时原生 TensorStore 线程阻塞在磁盘 `rq_qos_wait`，主线程等待设备到主机拷贝；终止进程后也曾因不可中断 I/O 暂时无法释放显存。最终已退出，未完成的临时检查点已清理，日志保留。没有保留这两轮的可用训练权重。

降低本进程 I/O 并发后，`scripts/verify_full_checkpoint_io.py` 在四个 CPU 设备上成功完成官方 π0.5 **完整训练状态**的保存与恢复，包括参数、优化器状态、EMA 和归一化资产。验证了恢复步数为 0、全部状态形状与类型匹配、代表性参数完全一致，总耗时约 149.85 秒。此测试不执行梯度更新，检查点是初始化状态，不能作为已训练权重使用。结果：`artifacts/formal_training/full_checkpoint_io_verification.json`。

验证检查点：`/data1/yanfeng/checkpoints/atombridge/local_mani_thermalcycler_pi05/pi05_base_fsdp4_cpu_io_20261119/1`，约 24 GiB。文件夹编号 1 是保存编号，内部实际优化器 step 为 0。

最终版小模型完整/部分/空权重初始化以及四设备优化器/EMA 保存恢复均通过，日志为 `artifacts/formal_training/fsdp_initialization_regression_unsandboxed.log`。

**剩余验证范围**：限流后的 GPU 端完整检查点保存尚未复测；诊断结束后其他任务已占用原先四卡中的 GPU 5、7，且用户明确要求暂不启动训练。因此不能宣称四卡端到端保存/恢复已验证或长期训练稳定性已验证。

## 正式训练命令

在项目根目录执行，两个场景依次训练，避免同时占用同四卡：

```bash
bash scripts/train_four_gpu.sh thermal --mode full
bash scripts/train_four_gpu.sh centrifuge --mode full
```

恢复同一正式实验：

```bash
bash scripts/train_four_gpu.sh thermal --mode full --resume
```

正式实验默认各 50,000 步，目录分别为：

- `/data1/yanfeng/checkpoints/atombridge/local_mani_thermalcycler_pi05/formal_50k`
- `/data1/yanfeng/checkpoints/atombridge/local_mani_centrifuge5910_pi05/formal_50k`

数据和权重仍通过项目软链接写入 `/data1/yanfeng`。本轮不自动启动 50,000 步正式实验。原生 TensorStore 和 GPU 操作在当前 Codex 沙箱内受限，验证使用沙箱外权限；普通终端可直接执行上述命令。
