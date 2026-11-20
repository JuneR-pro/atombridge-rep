"""Prepare normalization and local-weight training configs for the two formal datasets."""
import argparse
import faulthandler
import functools
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/formal_training"
ASSETS = ROOT / "data/assets"
INITIAL = ROOT.parent / "openpi/checkpoints/pi05_base/params"
REPOS = {"thermal": "mani_thermalcycler", "centrifuge": "mani_centrifuge5910"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temp.replace(path)


def offline_assets():
    from urllib.parse import urlparse
    from openpi.shared import download
    cache = ROOT / "artifacts/thermal_pilot5/openpi_cache"
    assert (cache / "big_vision/paligemma_tokenizer.model").is_file()
    original = download.maybe_download

    def local_only(url, **kwargs):
        parsed = urlparse(str(url))
        if parsed.scheme:
            target = cache / parsed.netloc / parsed.path.lstrip("/")
            if not target.exists():
                raise FileNotFoundError(f"Offline asset is not cached: {url}")
            return target.resolve()
        return original(str(url), **kwargs)

    download.maybe_download = local_only


def configure_checkpoint_io():
    # /data1 is a shared HDD. Orbax's default 128 file workers can overwhelm
    # its request queue while writing full optimizer/EMA checkpoints.
    # Limit this process's TensorStore pools without changing the file format.
    from orbax.checkpoint._src.serialization import tensorstore_utils
    tensorstore_utils.get_ts_context = functools.partial(
        tensorstore_utils.get_ts_context,
        file_io_concurrency_limit=2,
        data_copy_concurrency_limit=4,
    )


def build_config(scene, mode, exp_name="formal_50k", batch_size=None, fsdp_devices=None, resume=False):
    from openpi.training.config import AssetsConfig, get_config
    from openpi.training.weight_loaders import CheckpointWeightLoader
    assert INITIAL.is_dir(), INITIAL
    repo = REPOS[scene]
    original_name = f"{repo}_pi05" + ("-lora" if mode == "lora" else "")
    base = get_config(original_name)
    config = replace(base, name="local_" + original_name.replace("-", "_"), exp_name=exp_name,
                     data=replace(base.data, assets=AssetsConfig(assets_dir=str(ASSETS.resolve()), asset_id=repo)),
                     weight_loader=CheckpointWeightLoader(str(INITIAL)),
                     assets_base_dir=str(ROOT / "third_party/openpi/assets"),
                     checkpoint_base_dir=str((ROOT / "checkpoints").resolve()),
                     resume=resume)
    if batch_size is not None:
        config = replace(config, batch_size=batch_size)
    if fsdp_devices is not None:
        config = replace(config, fsdp_devices=fsdp_devices)
    else:
        config = replace(config, fsdp_devices=4)
    return config


class NumericDataset:
    """The same state/action chunks as LeRobot, without reading unrelated video pixels."""
    def __init__(self, repo, horizon):
        import pyarrow.parquet as pq
        root = Path(os.environ["HF_LEROBOT_HOME"]) / repo
        tables = [pq.read_table(p, columns=["state", "actions"]) for p in sorted(root.glob("data/*/*.parquet"))]
        self.ends = np.cumsum([len(t) for t in tables])
        self.states = np.concatenate([np.asarray(t["state"].to_pylist(), dtype=np.float32) for t in tables])
        self.actions = np.concatenate([np.asarray(t["actions"].to_pylist(), dtype=np.float32) for t in tables])
        self.offsets = np.arange(horizon)
        self.image = np.zeros((1, 1, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        index = int(index)
        end = self.ends[np.searchsorted(self.ends, index, side="right")]
        future = np.minimum(index + self.offsets, end - 1)
        return {"state": self.states[index], "actions": self.actions[future],
                "image": self.image, "wrist_image": self.image, "prompt": "normalization"}


def validate_numeric(config, data_config, numeric):
    from openpi import transforms
    from openpi.shared.normalize import RunningStats
    from openpi.training import data_loader
    original = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    # Include all task transitions and episode tails where future actions are padded.
    indices = set(range(96))
    for ep in (0, 99, 100, 199, 200, 299, 300, 399, 400, 499, 500, 599, 600, 699):
        start = 0 if ep == 0 else int(numeric.ends[ep - 1])
        end = int(numeric.ends[ep])
        indices.update((start, max(start, end - 50), end - 1))
    reference_stats = {key: RunningStats() for key in ("state", "actions")}
    numeric_stats = {key: RunningStats() for key in ("state", "actions")}
    buffers = {key: [] for key in reference_stats}
    for index in sorted(indices):
        expected = transform(original[index])
        actual = transform(numeric[index])
        for key in reference_stats:
            assert np.array_equal(expected[key], actual[key]), (index, key)
            if index < 96:
                buffers[key].append((expected[key], actual[key]))
    for key in reference_stats:
        for start in range(0, 96, 32):
            pairs = buffers[key][start:start + 32]
            reference_stats[key].update(np.stack([p[0] for p in pairs]))
            numeric_stats[key].update(np.stack([p[1] for p in pairs]))
        expected, actual = reference_stats[key].get_statistics(), numeric_stats[key].get_statistics()
        for field in ("mean", "std", "q01", "q99"):
            assert np.array_equal(getattr(expected, field), getattr(actual, field)), (key, field)
    result = {"samples_compared": len(indices), "all_tasks_and_episode_padding_checked": True,
              "state_actions_bitwise_equal": True, "first_three_batch_statistics_bitwise_equal": True}
    write_json(RUN / config.data.repo_id / "numeric_equivalence.json", result)
    return result


def prepare(scene):
    from openpi.shared import normalize
    from openpi.training import data_loader
    sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
    import compute_norm_stats
    config = build_config(scene, "full")
    folder = RUN / config.data.repo_id
    folder.mkdir(parents=True, exist_ok=True)
    data_config = config.data.create(config.assets_dirs, config.model)
    numeric = NumericDataset(config.data.repo_id, config.model.action_horizon)
    equivalence = validate_numeric(config, data_config, numeric)
    print(f"Numeric equivalence passed: {equivalence}", flush=True)
    # Retain the repo's batching, transforms, drop-last behavior and RunningStats.
    with patch.object(data_loader, "create_torch_dataset", return_value=numeric):
        loader, batches = compute_norm_stats.create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, num_workers=0)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    started = time.time()
    for index, batch in enumerate(loader):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
        if index % 100 == 0 or index == batches - 1:
            status = {"stage": "normalizing", "completed_batches": index + 1, "total_batches": batches,
                      "elapsed_seconds": time.time() - started}
            write_json(folder / "status.json", status)
            print(f"Normalization {index + 1}/{batches}", flush=True)
    output = ASSETS / config.data.repo_id
    result = {key: value.get_statistics() for key, value in stats.items()}
    for key, value in result.items():
        for name in ("mean", "std", "q01", "q99"):
            array = getattr(value, name)
            assert array.shape == (32,) and np.isfinite(array).all(), (key, name)
    normalize.save(output, result)
    # Expose the same stats to the original README full and LoRA config names.
    for mode in ("full", "lora"):
        local = build_config(scene, mode)
        name = f"{config.data.repo_id}_pi05" + ("-lora" if mode == "lora" else "")
        for config_name in (name, local.name):
            link = ROOT / "third_party/openpi/assets" / config_name / config.data.repo_id
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                assert link.resolve() == output.resolve(), link
            elif link.exists():
                raise FileExistsError(f"Refusing to replace assets: {link}")
            else:
                link.symlink_to(output.resolve(), target_is_directory=True)
        write_json(folder / f"{mode}_config.json", asdict(local))
        assert local.data.create(local.assets_dirs, local.model).norm_stats is not None
    status = {"stage": "complete", "total_frames": len(numeric), "batches": batches,
              "batch_size": config.batch_size, "frames_used": batches * config.batch_size,
              "dropped_tail_frames": len(numeric) % config.batch_size,
              "state_vectors": stats["state"]._count, "action_vectors": stats["actions"]._count,
              "output": str(output.resolve()), "elapsed_seconds": time.time() - started}
    write_json(folder / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("scene", choices=REPOS)
    parser.add_argument("--mode", choices=("full", "lora"), default="full")
    parser.add_argument("--exp-name", default="formal_50k")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fsdp-devices", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-train-steps", type=int, help="Override total steps for a separate short validation run")
    args = parser.parse_args()
    if timeout := os.environ.get("ATOMBRIDGE_TRACEBACK_TIMEOUT"):
        faulthandler.dump_traceback_later(int(timeout), repeat=True)
    offline_assets()
    configure_checkpoint_io()
    if args.stage == "prepare":
        prepare(args.scene)
    else:
        sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
        import train
        config = build_config(args.scene, args.mode, args.exp_name, args.batch_size, args.fsdp_devices, args.resume)
        if args.num_train_steps is not None:
            if args.num_train_steps < 1:
                parser.error("--num-train-steps must be positive")
            config = replace(config, num_train_steps=args.num_train_steps, log_interval=1)
        write_json(RUN / config.data.repo_id / f"launch_{args.exp_name}_{args.mode}.json", asdict(config))
        train.main(config)
    if os.environ.get("ATOMBRIDGE_TRACEBACK_TIMEOUT"):
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
