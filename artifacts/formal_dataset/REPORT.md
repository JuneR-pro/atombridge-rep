# 正式数据集准备与存储结果

日期：2026-11-18。状态：完成。使用已有环境、本地仿真轨迹，无模型或数据下载。

## 数据集

按照 README 使用 50 Hz、224×224 渲染，再转换为独立的 LeRobot 数据集。
每个场景 7 类任务，每类 100 条。原始视频包含三个视角；训练数据按仓库相机映射
保留前视 `image` 和腕部 `wrist_image`，状态和动作均为 7 维。

| 场景 | Repo ID | 演示数 | 帧数 | 完成标签正例帧 | 数据集文件大小 |
| --- | --- | --- | --- | --- | --- |
| 热循环仪 | `mani_thermalcycler` | 700 | 349,419 | 36,769 | 0.81 GiB |
| 离心机 | `mani_centrifuge5910` | 700 | 564,505 | 46,758 | 1.78 GiB |

共 1400 条演示、913,924 帧。转换后全部属于仓库默认 train split，本次没有额外划分
验证集或测试集，也没有训练或评测策略。

## 存储

实际存储位置：

- `/data1/yanfeng/datasets/atombridge/lerobot/mani_thermalcycler`
- `/data1/yanfeng/datasets/atombridge/lerobot/mani_centrifuge5910`
- 原始轨迹与渲染视频：`/data1/yanfeng/datasets/atombridge/raw/`
- 现有与后续训练 checkpoint：`/data1/yanfeng/checkpoints/atombridge/`

项目的 `data`、`checkpoints`、两个原始轨迹目录均为软链接。
`third_party/openpi/checkpoints` 也链接到项目根目录的 `checkpoints`，因此 README
从 OpenPI 子目录启动的默认训练输出会写到 `/data1`。此前 pilot 数据和 3 步训练
checkpoint 已一并迁移，原路径仍可访问；本次没有生成正式训练权重。

激活脚本默认设置 `HF_LEROBOT_HOME` 为项目 `data/lerobot`，后续训练读取同一目录。
显式指定其他 checkpoint 输出位置的命令仍需自行使用上述目录。

## 转换与检查

使用 `scripts/prepare_formal_dataset.py` 调用原始渲染函数；渲染结果有逐条完成标记。
最初离心机使用 GPU 2，确认 GPU 3 更快后保留完成结果并切换到 GPU 3。
两个场景的全部渲染均已完成。

为减少逐帧统计和重编码的总耗时，使用 `scripts/parallel_convert.py` 将每个场景按原始
排序分成 7 个 100 条的分块，4 路调用未修改的 `scripts/convert.py`。随后按原顺序合并，
重编号全局帧、episode 和 task 索引，并使用 LeRobot 原生元数据 API 合并统计。
合并视频使用硬链接保留原编码内容，不重新编码。

先对两个不同任务的 1283 帧样本比较串行与分块结果：Parquet 所有数值相同，视频
SHA-256 相同，元数据和统计相同，统计最大绝对差为 0，且 LeRobot 原生解码通过。
结果见 `parallel_validation.json`。

全量检查覆盖全部 1400 条轨迹：

- 每个场景 700 条、每类 100 条，帧数与原始降采样记录一致。
- 状态、动作逐值等于原始轨迹相应索引转换得到的 float32 值，且全部有限。
- 完成标签逐帧等于原始记录，任务索引与原始语言提示对应。
- 全局帧索引连续，episode 索引与轨迹排序一致。
- 视频逐条通过原转换脚本的解码及非黑屏检查；最终视频帧数、分辨率和帧率一致。
- 两个数据集均通过 LeRobot 原生加载，并抽查首部、中部、末尾帧：状态 `[7]`，
  动作序列 `[50,7]`，两路图像 `[3,224,224]`，包含末尾动作补齐情况。

检查明细：`thermal/dataset_audit.json`、`centrifuge/dataset_audit.json`，
原生读取结果为各子目录的 `training_input_check.json`。
`manifest.json` 记录最终路径、数量、文件大小、软链接与源文件校验和。

分块结果及日志保留在 `/data1/yanfeng/datasets/atombridge/conversion_work/`；
先前串行转换的中间结果保留在其 `thermal_serial_partial/` 子目录。

## 下一步

计算这两个正式数据集的归一化统计，配置现有本地初始化权重，然后开始训练。
当前完成数据准备，尚未进行正式训练或论文指标评测。
