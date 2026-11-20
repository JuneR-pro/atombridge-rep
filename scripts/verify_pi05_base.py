"""Verify the official download can initialize this repo's full pi05 model."""
from collections import Counter
import faulthandler
import json
from pathlib import Path
import sys
import time

from flax import nnx
import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/pi05_base"
TARGET = ROOT.parent / "openpi/checkpoints/pi05_base"
sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
import train
from openpi.training.config import get_config
from openpi.training.weight_loaders import CheckpointWeightLoader

status = json.loads((RUN / "download_status.json").read_text())
assert status["stage"] == "complete"
assert len(json.loads((RUN / "verified_files.json").read_text())) == 29
config = get_config("mani_thermalcycler_pi05")
faulthandler.dump_traceback_later(30, repeat=True)
print("Inferring model parameter shapes", flush=True)
shapes = jax.eval_shape(lambda key: nnx.state(config.model.create(key)).to_pure_dict(), jax.random.key(42))
print("Restoring and validating official checkpoint", flush=True)
started = time.time()
params = train._load_weights_and_validate(CheckpointWeightLoader(str(TARGET / "params")), shapes)
arrays = jax.tree.leaves(params)
print("Checking finite parameter values", flush=True)
assert all(np.isfinite(x).all() for x in arrays)
result = {"status": "complete", "path": str(TARGET), "source": "gs://openpi-assets/checkpoints/pi05_base",
          "files": 29, "download_bytes": status["total_bytes"], "crc32c_all_passed": True,
          "model_parameter_shapes_and_dtypes_match": True, "all_parameters_finite": True,
          "parameter_arrays": len(arrays), "parameter_elements": int(sum(x.size for x in arrays)),
          "in_memory_bytes": int(sum(x.nbytes for x in arrays)), "dtypes": dict(Counter(str(x.dtype) for x in arrays)),
          "restore_and_check_seconds": time.time() - started, "training_steps": 0}
(RUN / "checkpoint_verification.json").write_text(json.dumps(result, indent=2) + "\n")
faulthandler.cancel_dump_traceback_later()
print(json.dumps(result, indent=2))
