# README 数据收集结果

日期：2026-11-18。状态：完成，两个采集命令和两次全量检查均正常退出。

按照 README 的 Training Data Generation 部分，使用现有虚拟环境和本地依赖，
以 CPU 并行运行两组仓库原始专家脚本。未下载模型或数据，未修改专家动作、成功判据、
场景、任务列表和采集种子规则。之前的 pilot 数据与 checkpoint 均保留。

| 场景 | 原子任务类别 | 每类成功演示 | 总演示 | 原始仿真状态数 | 文件总大小 |
| --- | --- | --- | --- | --- | --- |
| 热循环仪 | 7 | 100 | 700 | 3,491,112 | 1.90 GiB |
| 离心机 | 7 | 100 | 700 | 5,641,793 | 3.67 GiB |

原始轨迹位置：

- `logs/thermal_cycler_tasks/`
- `logs/centrifuge5910_tasks/`

每个场景包含共享 `model.mjb`，每条轨迹包含 `model.xml`、`states.npy.zst`、
`info.json`。原始状态数不是渲染后的训练帧数。

## 命令与重试

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  bash scripts/atombridge_python.sh -u scripts/autobio_scripts/thermal_cycler_tasks.py \
  --episodes 100 --log-root logs/thermal_cycler_tasks

JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  bash scripts/atombridge_python.sh -u scripts/autobio_scripts/centrifuge5910_tasks.py \
  --episodes 100 --log-root logs/centrifuge5910_tasks
```

每类编号为 0–99，首次种子等于编号。使用仓库默认最多 3 次尝试，重试种子为
`编号 + 尝试次数 * 100`。热循环仪全部首次成功；离心机关盖任务编号 15 首次失败，
第二次使用种子 215 成功。仓库脚本已自动删除失败轨迹，最终保留 1400 条成功演示。

## 全量检查

通过 `scripts/audit_raw_collection.py` 对全部 1400 条轨迹解压检查：

- 14 类任务各 100 条，编号完整且无重复。
- 共享模型、轨迹 XML、压缩状态与 JSON 注释齐全且非空。
- 仿真状态为 float64，维度与元数据一致，全部数值有限。
- 每条轨迹的时间严格递增，状态行数与注释数量一致。
- 策略使用的状态和动作索引均为 7 维。
- 完成标签均为布尔值，所有轨迹最终标签均为成功。
- 采集日志记录的成功次数、唯一任务编号与检查结果一致。

逐条结果见 `thermal_audit.json` 和 `centrifuge_audit.json`。
`manifest.json` 记录命令、版本、种子、重试、文件大小与源文件/模型校验和。
采集日志为 `thermal.log`、`centrifuge.log`。

## 下一步

按照 README 进行 50 Hz 相机渲染，再转换为 LeRobot 数据集。当前完成的是原始专家
轨迹收集，尚未进行这一批数据的渲染、正式训练或策略成功率评测，因此不能据此声称
已复现论文结果。后续渲染优先使用之前验证可运行的单 worker 设置。

不要直接对相同输出目录重跑完整采集命令：仓库没有断点续采参数，会创建新后缀目录
并产生重复编号。需要重新采集时应使用新的输出目录。
