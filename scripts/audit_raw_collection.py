"""Audit locally generated expert episodes without changing their contents."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re

import numpy as np
import zstandard as zstd


def audit(root, expected):
    counts = Counter()
    indices = {}
    rows = []
    errors = []
    if not (root / "model.mjb").is_file():
        errors.append("Missing shared model.mjb")
    for episode in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            match = re.fullmatch(r"(.+)_(\d{4})_(\d{4})", episode.name)
            assert match, "Unexpected episode directory name"
            task, index, _ = match.groups()
            index = int(index)
            assert index not in indices.setdefault(task, set()), "Duplicate episode index"
            indices[task].add(index)
            for name in ("model.xml", "states.npy.zst", "info.json"):
                assert (episode / name).stat().st_size > 0, f"Missing or empty {name}"
            meta = json.loads((episode / "info.json").read_text())
            with (episode / "states.npy.zst").open("rb") as source:
                with zstd.ZstdDecompressor().stream_reader(source) as reader:
                    states = np.load(io.BytesIO(reader.read()), allow_pickle=False)
            assert states.ndim == 2 and len(states) > 1, "Invalid states shape"
            assert states.dtype == np.float64, "Unexpected states dtype"
            assert states.shape[1] == max(s["end"] for s in meta["split"].values()), "Invalid state width"
            assert np.isfinite(states).all(), "Nonfinite physics states"
            assert len(meta["info"]) == len(states), "State/annotation length mismatch"
            assert len(meta["task"]["state_indices"]) == 7, "Unexpected policy state dimension"
            assert len(meta["task"]["action_indices"]) == 7, "Unexpected policy action dimension"
            time_split = meta["split"]["time"]
            times = states[:, time_split["start"]:time_split["end"]].ravel()
            assert np.all(np.diff(times) > 0), "Nonincreasing simulation time"
            labels = [f["task_is_complete"] for f in meta["info"]]
            assert all(type(label) is bool for label in labels), "Invalid completion label"
            assert labels[-1], "Final state does not pass task completion check"
            rows.append({"episode": episode.name, "task": task, "index": index,
                         "states": len(states), "positive_states": sum(labels),
                         "duration_seconds": float(times[-1] - times[0])})
            counts[task] += 1
        except Exception as exc:
            errors.append(f"{episode.name}: {type(exc).__name__}: {exc}")
    if len(counts) != 7:
        errors.append(f"Expected 7 tasks, found {len(counts)}")
    for task, found in indices.items():
        if found != set(range(expected)):
            errors.append(f"{task}: indices differ from 0..{expected - 1}")
        if counts[task] != expected:
            errors.append(f"{task}: expected {expected} valid episodes, got {counts[task]}")
    return {"root": str(root), "episodes": len(rows), "counts": dict(counts),
            "states": sum(row["states"] for row in rows),
            "positive_states": sum(row["positive_states"] for row in rows),
            "errors": errors, "episode_details": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episode_details"}, indent=2))
    raise SystemExit(bool(result["errors"]))
