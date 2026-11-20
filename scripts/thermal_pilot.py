"""Local, offline thermal-cycler pilot: render, normalize, and smoke-train."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/thermal_pilot5"
RAW = ROOT / "logs/thermal_cycler_pilot5"
REPO = "atombridge_thermal_pilot5"
INITIAL_PARAMS = Path("/data1/yanfeng/models/robocasa365/pi05_pretrain_human300/multitask_learning/75000/params")


class MirrorStream:
    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, value):
        self.console.write(value)
        if not self.logfile.closed:
            self.logfile.write(value)
        return len(value)

    def flush(self):
        if not self.console.closed:
            self.console.flush()
        if not self.logfile.closed:
            self.logfile.flush()

    def close(self):
        self.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)

    def isatty(self):
        return False


def write_json(name, value):
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def prepare_offline_assets():
    from urllib.parse import urlparse
    from openpi.shared import download

    cache = RUN / "openpi_cache"
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer.exists():
        source = Path("/home/yanfeng/.cache/openpi/big_vision/paligemma_tokenizer.model")
        shutil.copyfile(source, tokenizer)
    os.environ["OPENPI_DATA_HOME"] = str(cache)
    original = download.maybe_download

    def local_only(url, **kwargs):
        parsed = urlparse(str(url))
        if parsed.scheme:
            path = cache / parsed.netloc / parsed.path.lstrip("/")
            if not path.exists():
                raise FileNotFoundError(f"Offline pilot: uncached remote asset {url}")
            return path.resolve()
        return original(str(url), **kwargs)

    download.maybe_download = local_only


def build_config(args):
    from openpi.training.config import AssetsConfig, get_config
    from openpi.training.weight_loaders import CheckpointWeightLoader

    if not INITIAL_PARAMS.is_dir():
        raise FileNotFoundError(INITIAL_PARAMS)
    base = get_config("mani_thermalcycler_pi05-lora")
    return replace(
        base,
        name="local_thermal_pilot5_pi05_lora",
        exp_name=args.exp_name,
        data=replace(base.data, repo_id=REPO, assets=AssetsConfig(asset_id=REPO)),
        weight_loader=CheckpointWeightLoader(str(INITIAL_PARAMS)),
        assets_base_dir=str(RUN / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints/thermal_pilot5"),
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=args.steps,
        log_interval=1,
        save_interval=args.steps,
        keep_period=None,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    )


def render(args):
    episodes = sorted(p.parent for p in RAW.glob("*/states.npy.zst"))
    if len(episodes) != 35:
        raise ValueError(f"Expected 35 episodes, found {len(episodes)}")
    logs = RUN / "render_logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def render_one(episode):
        marker = episode / "render_50hz_complete.json"
        if marker.exists():
            return episode.name, "already_complete"
        env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                   CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_EGL_DEVICE_ID=str(args.gpu))
        command = [sys.executable, str(ROOT / "scripts/autobio_scripts/render.py"),
                   str(episode), "--height", "224", "--width", "224", "--fps", "50"]
        with (logs / f"{episode.name}.log").open("w") as output:
            subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        metadata = json.loads((episode / "downsample.json").read_text())
        assert metadata["fps"] == 50
        for filename in metadata["cameras"].values():
            if not (episode / filename).is_file():
                raise FileNotFoundError(episode / filename)
        marker.write_text(json.dumps({"fps": 50, "frames": len(metadata["indices"])}) + "\n")
        return episode.name, "complete"

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_one, episode) for episode in episodes]
        for future in as_completed(futures):
            name, state = future.result()
            results[name] = state
            write_json("render_status.json", {"completed": len(results), "total": len(episodes),
                       "elapsed_seconds": time.time() - started, "episodes": results})
            print(f"Rendered {len(results)}/{len(episodes)}: {name}", flush=True)


def normalize(args):
    import numpy as np
    from openpi.shared import normalize as normalization

    sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
    import compute_norm_stats

    prepare_offline_assets()
    config = build_config(args)
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, num_batches = compute_norm_stats.create_torch_dataloader(
        data_config, config.model.action_horizon, batch_size=32,
        model_config=config.model, num_workers=2,
    )
    stats = {key: normalization.RunningStats() for key in ("state", "actions")}
    for index, batch in enumerate(loader):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
        if index % 25 == 0:
            print(f"Normalization batch {index + 1}/{num_batches}", flush=True)
    output = config.assets_dirs / REPO
    normalization.save(output, {key: value.get_statistics() for key, value in stats.items()})
    write_json("normalization.json", {"batches": num_batches, "output": output})
    print(f"Normalization saved: {output}", flush=True)


def train(args):
    sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
    import train as openpi_train

    prepare_offline_assets()
    config = build_config(args)
    write_json("train_config.json", asdict(config))
    logging.basicConfig(level=logging.INFO)
    openpi_train.main(config)
    write_json("train_result.json", {"status": "complete", "steps": args.steps,
               "checkpoint_dir": config.checkpoint_dir, "initial_params": INITIAL_PARAMS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["render", "stats", "train"])
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--exp-name", default="smoke3")
    args = parser.parse_args()
    os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "data/lerobot"))
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / f"{args.stage}_{args.exp_name}.log").open("a") as logfile:
        with redirect_stdout(MirrorStream(sys.stdout, logfile)), redirect_stderr(MirrorStream(sys.stderr, logfile)):
            {"render": render, "stats": normalize, "train": train}[args.stage](args)


if __name__ == "__main__":
    main()
