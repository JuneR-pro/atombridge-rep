"""Verify full pi05_base train-state save/restore on four CPU devices; no training updates."""
import json
import sys
import time
from types import SimpleNamespace

import jax
import numpy as np

from formal_training import ROOT, build_config, configure_checkpoint_io, offline_assets, write_json

sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
import train
from openpi.training import sharding

assert jax.default_backend() == "cpu" and jax.device_count() == 4
offline_assets()
configure_checkpoint_io()
config = build_config("thermal", "full", "pi05_base_fsdp4_cpu_io_20261119")
train.init_logging()
mesh = sharding.make_mesh(4)
started = time.time()
state, _ = train.init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
jax.block_until_ready(state)
print("Full CPU train state initialized", flush=True)
manager, _ = train._checkpoints.initialize_checkpoint_dir(
    config.checkpoint_dir, keep_period=None, overwrite=False, resume=False
)
loader = SimpleNamespace(data_config=lambda: config.data.create(config.assets_dirs, config.model))
train._checkpoints.save_state(manager, state, loader, 1)
manager.wait_until_finished()
print("Full CPU checkpoint committed", flush=True)
abstract_state, _ = train.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
restored = train._checkpoints.restore_state(manager, abstract_state, loader, 1)
jax.block_until_ready(restored)
assert int(restored.step) == 0
before = state.params.to_pure_dict()["action_out_proj"]["kernel"]
after = restored.params.to_pure_dict()["action_out_proj"]["kernel"]
np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
assert restored.ema_params is not None
for original, loaded in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
    assert original.shape == loaded.shape and original.dtype == loaded.dtype
manager.close()
result = {"status": "complete", "backend": "cpu", "devices": 4,
          "checkpoint": str(config.checkpoint_dir / "1"), "executed_training_steps": 0,
          "full_state_save_restore_passed": True, "step_restored": 0,
          "representative_parameter_exactly_equal": True,
          "all_state_shapes_and_dtypes_match": True, "ema_restored": True,
          "file_io_concurrency_limit": 2, "seconds": time.time() - started}
write_json(ROOT / "artifacts/formal_training/full_checkpoint_io_verification.json", result)
print(json.dumps(result, indent=2), flush=True)
