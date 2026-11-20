"""Run the unchanged repo converter on shards, then merge LeRobot v2 metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def merge(shards, output, repo_id):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import write_info

    first = LeRobotDatasetMetadata(shards[0].name, root=shards[0])
    target = LeRobotDatasetMetadata.create(repo_id, fps=first.fps, root=output,
                                           features=first.features, robot_type=first.robot_type)
    for shard in shards:
        source = LeRobotDatasetMetadata(shard.name, root=shard)
        assert source.fps == target.fps and source.features == target.features
        for local_index in range(source.total_episodes):
            episode = source.episodes[local_index]
            episode_index = target.total_episodes
            tasks = episode["tasks"]
            for task in tasks:
                if target.get_task_index(task) is None:
                    target.add_task(task)
            table = pq.read_table(shard / source.get_data_file_path(local_index))
            task_map = {key: target.get_task_index(value) for key, value in source.tasks.items()}
            replacements = {
                "episode_index": np.full(len(table), episode_index, dtype=np.int64),
                "index": np.arange(target.total_frames, target.total_frames + len(table), dtype=np.int64),
                "task_index": np.asarray([task_map[i] for i in table["task_index"].to_pylist()], dtype=np.int64),
            }
            stats = dict(source.episodes_stats[local_index])
            for key, values in replacements.items():
                field = table.schema.field(key)
                table = table.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
                stats[key] = get_feature_stats(values, axis=0, keepdims=True)
            parquet = output / target.get_data_file_path(episode_index)
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet)
            for key in target.video_keys:
                video = output / target.get_video_file_path(episode_index, key)
                video.parent.mkdir(parents=True, exist_ok=True)
                # Same filesystem: the merged dataset owns an independent hard link.
                # Shard cleanup therefore cannot remove the merged video contents.
                os.link(shard / source.get_video_file_path(local_index, key), video)
            target.save_episode(episode_index, len(table), tasks, stats)
    target.update_video_info()
    write_info(target.info, target.root)
    return {"episodes": target.total_episodes, "frames": target.total_frames,
            "tasks": len(target.tasks), "shards": len(shards)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=100)
    args = parser.parse_args()
    assert args.workers > 0 and args.shard_size > 0
    home = Path(os.environ["HF_LEROBOT_HOME"])
    output = home / args.repo_id
    if output.exists():
        raise FileExistsError(f"Refusing to replace dataset: {output}")
    work = home.parent / "conversion_work" / args.repo_id
    work.mkdir(parents=True, exist_ok=True)
    episodes = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    assert episodes and all((p / "render_50hz_complete.json").exists() for p in episodes)
    groups = [episodes[i:i + args.shard_size] for i in range(0, len(episodes), args.shard_size)]

    def convert_shard(item):
        index, group = item
        name = f"shard_{index:03d}"
        raw = work / "raw" / name
        raw.mkdir(parents=True, exist_ok=True)
        for episode in group:
            link = raw / episode.name
            if not link.exists():
                link.symlink_to(episode.resolve(), target_is_directory=True)
        destination = work / "datasets" / name
        marker = work / f"{name}_complete.json"
        if not marker.exists():
            if destination.exists():
                raise FileExistsError(f"Incomplete shard retained for inspection: {destination}")
            env = dict(os.environ, HF_LEROBOT_HOME=str(work / "datasets"))
            with (work / f"{name}.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/convert.py"),
                                "--data-dir", str(raw), "--repo-id", name],
                               env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            marker.write_text(json.dumps({"episodes": [p.name for p in group]}) + "\n")
        else:
            assert json.loads(marker.read_text())["episodes"] == [p.name for p in group]
        print(f"Converted shard {index + 1}/{len(groups)} ({len(group)} episodes)", flush=True)
        return destination

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        shards = list(pool.map(convert_shard, enumerate(groups)))
    result = merge(shards, output, args.repo_id)
    (work / "merge_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
