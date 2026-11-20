# pi05_base 下载与加载验证

- 实际目录：`/home/yanfeng/code/EmA/openpi/checkpoints/pi05_base`
- 官方源：`gs://openpi-assets/checkpoints/pi05_base`，见 [OpenPI](https://github.com/Physical-Intelligence/openpi)。
- 29 个文件，共 12,441,749,581 字节；全部通过官方 CRC32C 校验，并记录 SHA256。
- `gcs_manifest.json` 保存官方对象 generation、大小和校验值；`verified_files.json` 保存本地验证结果。
- `checkpoint_verification.json` 记录本项目原生 CheckpointWeightLoader 的加载验证：51 个参数数组、3,353,433,872 个参数，全部 float32，形状和类型匹配，数值全部有限。恢复及数值检查约 8.65 秒，执行训练步数 0。
- 沙箱内原生 TensorStore 元数据读取曾阻塞；在沙箱外运行相同验证脚本成功。后续相同读取操作可能需要相同执行权限。
- `scripts/download_pi05_base.py` 支持重复运行、已完成文件校验与分块断点续传；本次下载现已全部完成。

2026-11-19：训练入口已切换到上述官方 pi05_base，默认使用 FSDP 4；初始化输入分片修复已应用到实际训练器。短程验证记录见 `artifacts/four_gpu_assessment/TRAINING_READINESS.md`。
