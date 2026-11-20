"""Render one collected scene, convert with the repo converter, and audit it."""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = {"thermal": ("thermal_cycler_tasks", "mani_thermalcycler"),
          "centrifuge": ("centrifuge5910_tasks", "mani_centrifuge5910")}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def audit_dataset(raw, dataset):
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import zstandard as zstd
    info = json.loads((dataset / "meta/info.json").read_text())
    task_indices = {item["task"]: item["task_index"] for item in
                    (json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines())}
    assert info["fps"] == 50 and info["total_episodes"] == 700
    raw_episodes = sorted(p.parent for p in raw.glob("*/states.npy.zst"))
    parquet_files = sorted(dataset.glob("data/*/*.parquet"))
    assert len(parquet_files) == len(raw_episodes) == 700
    frames = positive = 0
    task_counts = Counter()
    for index, (source, parquet) in enumerate(zip(raw_episodes, parquet_files)):
        meta = json.loads((source / "info.json").read_text())
        downsample = json.loads((source / "downsample.json").read_text())
        table = pq.read_table(parquet).to_pydict()
        state, action = np.asarray(table["state"]), np.asarray(table["actions"])
        labels = np.asarray(table["task_is_complete"]).reshape(-1)
        count = len(downsample["indices"])
        assert state.shape == action.shape == (count, 7)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        with (source / "states.npy.zst").open("rb") as stream:
            with zstd.ZstdDecompressor().stream_reader(stream) as reader:
                original = np.load(io.BytesIO(reader.read()), allow_pickle=False)
        selected = original[downsample["indices"]]
        for key, values, indices_key in (("qpos", state, "state_indices"),
                                          ("ctrl", action, "action_indices")):
            split = meta["split"][key]
            expected_values = selected[:, split["start"]:split["end"]][:, meta["task"][indices_key]]
            assert np.array_equal(values, expected_values.astype(np.float32)), f"{key} differs: {source}"
        assert np.isin(labels, [0, 1]).all()
        expected = np.asarray([meta["info"][i]["task_is_complete"] for i in downsample["indices"]])
        assert np.array_equal(labels, expected), f"Completion labels differ: {source}"
        assert set(table["episode_index"]) == {index}
        assert set(table["task_index"]) == {task_indices[meta["task"]["prefix"]]}
        assert np.array_equal(table["index"], np.arange(frames, frames + count))
        assert np.array_equal(table["frame_index"], np.arange(count))
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            video = dataset / info["video_path"].format(episode_chunk=index // info["chunks_size"],
                                                       video_key=key, episode_index=index)
            cap = cv2.VideoCapture(str(video))
            assert cap.isOpened(), str(video)
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == count, str(video)
            assert abs(cap.get(cv2.CAP_PROP_FPS) - 50) < 0.01
            assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 224
            cap.release()
        task_counts[meta["task"]["prefix"]] += 1
        frames += count
        positive += int(labels.sum())
    assert info["total_frames"] == frames
    assert len(task_counts) == 7 and set(task_counts.values()) == {100}
    return {"episodes": 700, "frames": frames, "fps": 50, "resolution": [224, 224],
            "state_dimension": 7, "action_dimension": 7, "positive_frames": positive,
            "task_episode_counts": dict(task_counts), "all_finite": True,
            "labels_match_raw": True, "state_actions_match_raw": True,
            "frame_indices_contiguous": True, "video_frame_counts_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    scene, repo = SCENES[args.scene]
    raw = ROOT / "logs" / scene
    dataset = ROOT / "data/lerobot" / repo
    run = ROOT / "artifacts/formal_dataset" / args.scene
    run.mkdir(parents=True, exist_ok=True)
    status = {"stage": "render", "completed": 0, "total": 700, "gpu": args.gpu,
              "raw": str(raw.resolve()), "dataset": str(dataset.resolve())}
    started = time.time()
    try:
        os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                          MUJOCO_EGL_DEVICE_ID=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu))
        sys.path.insert(0, str(ROOT / "scripts/autobio_scripts"))
        import render
        import mujoco
        mujoco.mj_loadPluginLibrary(str(ROOT / "libmjlab.so.3.3.0"))
        episodes = sorted(p.parent for p in raw.glob("*/states.npy.zst"))
        assert len(episodes) == 700
        (run / "render_logs").mkdir(exist_ok=True)
        for index, episode in enumerate(episodes):
            marker = episode / "render_50hz_complete.json"
            if not marker.exists():
                with (run / "render_logs" / f"{episode.name}.log").open("w") as logfile:
                    with redirect_stdout(logfile), redirect_stderr(logfile):
                        render.render_mujoco_from_log(episode, height=224, width=224, fps=50)
                downsample = json.loads((episode / "downsample.json").read_text())
                assert downsample["fps"] == 50
                assert all((episode / name).stat().st_size > 0 for name in downsample["cameras"].values())
                write_json(marker, {"fps": 50, "frames": len(downsample["indices"])})
            status.update(completed=index + 1, elapsed_seconds=time.time() - started)
            write_json(run / "status.json", status)
            print(f"Rendered {index + 1}/700: {episode.name}", flush=True)
        status["stage"] = "convert"
        write_json(run / "status.json", status)
        if not (run / "conversion_complete.json").exists():
            if dataset.exists():
                raise FileExistsError(f"Refusing to overwrite partial or existing dataset: {dataset}")
            env = dict(os.environ, HF_LEROBOT_HOME=str(ROOT / "data/lerobot"))
            with (run / "convert.log").open("w") as logfile:
                subprocess.run([sys.executable, str(ROOT / "scripts/parallel_convert.py"),
                                "--data-dir", str(raw), "--repo-id", repo, "--workers", "4"],
                               env=env, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT, check=True)
            write_json(run / "conversion_complete.json", {"dataset": str(dataset.resolve())})
        status["stage"] = "audit"
        write_json(run / "status.json", status)
        result = audit_dataset(raw, dataset)
        write_json(run / "dataset_audit.json", result)
        status.update(stage="complete", elapsed_seconds=time.time() - started, audit=result)
        write_json(run / "status.json", status)
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as exc:
        status.update(stage="failed", error=repr(exc), elapsed_seconds=time.time() - started)
        write_json(run / "status.json", status)
        raise


if __name__ == "__main__":
    main()
