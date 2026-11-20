"""Exercise full, partial (LoRA-like), and empty checkpoint initialization on four CPU devices."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from formal_training import configure_checkpoint_io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
import train
from openpi.training.config import get_config
from openpi.training import sharding


class TinyModel(nnx.Module):
    def __init__(self):
        self.kernel = nnx.Param(jnp.ones((1024, 2048), jnp.float32))
        self.lora = nnx.Param(jnp.zeros((16, 1024), jnp.float32))


class TinyConfig:
    def create(self, rng):
        return TinyModel()


class Loader:
    def __init__(self, kind):
        self.kind = kind

    def load(self, shapes):
        return {
            key: np.full(value.shape, 3.25, value.dtype)
            if self.kind == "full" or (self.kind == "partial" and key == "kernel") else value
            for key, value in shapes.items()
        }


assert jax.device_count() == 4
configure_checkpoint_io()
mesh = sharding.make_mesh(4)
for kind in ("full", "partial", "empty"):
    config = replace(get_config("mani_thermalcycler_pi05"), model=TinyConfig(),
                     fsdp_devices=4, weight_loader=Loader(kind))
    state, expected_sharding = train.init_train_state(config, jax.random.key(42), mesh, resume=False)
    jax.block_until_ready(state)
    values = state.params.to_pure_dict()
    np.testing.assert_array_equal(np.asarray(values["kernel"]), 1.0 if kind == "empty" else 3.25)
    np.testing.assert_array_equal(np.asarray(values["lora"]), 3.25 if kind == "full" else 0.0)
    assert values["kernel"].sharding == expected_sharding.params.to_pure_dict()["kernel"]
    assert values["kernel"].addressable_shards[0].data.shape != values["kernel"].shape
    assert state.ema_params is not None
    print(f"PASS: {kind} checkpoint, parameter values, EMA and four-device sharding", flush=True)

with tempfile.TemporaryDirectory(prefix="atombridge-fsdp-") as folder:
    manager, _ = train._checkpoints.initialize_checkpoint_dir(
        Path(folder) / "checkpoints", keep_period=None, overwrite=False, resume=False
    )
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
    train._checkpoints.save_state(manager, state, loader, 1)
    manager.wait_until_finished()
    restored = train._checkpoints.restore_state(manager, state, loader, 1)
    np.testing.assert_array_equal(np.asarray(restored.params.to_pure_dict()["kernel"]), 1.0)
    manager.close()
    print("PASS: four-device optimizer/EMA checkpoint save and restore", flush=True)
