"""Assess the exact repo train-state sharding and optionally compile a GPU train step."""
import argparse
from dataclasses import replace
import functools
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/four_gpu_assessment"
sys.path.insert(0, str(ROOT / "third_party/openpi/scripts"))
import train
from openpi.training import config as configs, sharding
from openpi.shared import array_typing as at


def byte_count(tree):
    return int(sum(np.prod(x.shape) * np.dtype(x.dtype).itemsize
                   for x in jax.tree.leaves(tree) if hasattr(x, "dtype")))


def local_bytes(tree, shardings):
    sizes = jax.tree.map(lambda x, s: int(np.prod(s.shard_shape(x.shape)) * np.dtype(x.dtype).itemsize),
                         tree, shardings)
    return int(sum(jax.tree.leaves(sizes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "lora"), default="full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-init", action="store_true")
    parser.add_argument("--shard-initial-params", action="store_true")
    args = parser.parse_args()
    name = "mani_thermalcycler_pi05" + ("-lora" if args.mode == "lora" else "")
    config = replace(configs.get_config(name), batch_size=args.batch_size, fsdp_devices=args.fsdp_devices)
    if args.no_ema:
        config = replace(config, ema_decay=None)
    assert jax.device_count() == 4, jax.devices()
    mesh = sharding.make_mesh(args.fsdp_devices)
    rng = jax.random.key(config.seed)
    # resume=True returns only abstract shapes and sharding, without checkpoint reads or parameter allocation.
    state, state_sharding = train.init_train_state(config, rng, mesh, resume=True)
    groups = {}
    for key in ("params", "opt_state", "ema_params"):
        value = getattr(state, key)
        groups[key] = {"global_bytes": byte_count(value),
                       "per_device_bytes": local_bytes(value, getattr(state_sharding, key))}
    result = {"mode": args.mode, "batch_size_global": args.batch_size, "devices": [str(d) for d in jax.devices()],
              "backend": jax.default_backend(), "fsdp_devices": args.fsdp_devices, "ema_decay": config.ema_decay,
              "parameter_elements": int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params))),
              "trainable_elements": int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params.filter(config.trainable_filter)))),
              "state_groups": groups, "per_device_persistent_bytes": local_bytes(state, state_sharding)}
    suffix = f"{args.mode}_b{args.batch_size}_fsdp{args.fsdp_devices}" + ("_noema" if args.no_ema else "")
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / f"{suffix}_{jax.default_backend()}.json"
    if args.compile_init:
        output = output.with_name(output.stem + "_init_shapes.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if args.compile_init:
        original_jit = jax.jit
        param_sharding = state_sharding.params.to_pure_dict()

        def intercept(fun, *positional, **kwargs):
            if fun.__qualname__ != "init_train_state.<locals>.init" or "out_shardings" not in kwargs:
                return original_jit(fun, *positional, **kwargs)
            if args.shard_initial_params:
                kwargs["in_shardings"] = (
                    jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), param_sharding
                )

            def abstract_call(*arguments):
                with at.disable_typechecking():
                    compiled = original_jit(fun, *positional, **kwargs).lower(*arguments).compile()
                memory = compiled.memory_analysis()
                fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
                result["init_compiled_memory"] = {field: int(getattr(memory, field)) for field in fields}
                result["init_compiled_peak_buffer_estimate_bytes"] = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                                                                      + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
                result["initial_params_sharded"] = isinstance(kwargs["in_shardings"], tuple)
                return state
            return abstract_call

        with patch.object(jax, "jit", intercept), patch.object(train, "_load_weights_and_validate", lambda _, shapes: shapes):
            train.init_train_state(config, rng, mesh, resume=False)
        init_output = output.with_name(output.stem + ("_sharded_init" if result["initial_params_sharded"] else "_original_init") + ".json")
        init_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if args.compile:
        started = time.time()
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # JAX lower() reconstructs the state with ArgInfo placeholders, which are
        # not runtime arrays and must bypass the dataclass's runtime type checker.
        with at.disable_typechecking(), sharding.set_mesh(mesh):
            compiled = jax.jit(functools.partial(train.train_step, config),
                               in_shardings=(replicated, state_sharding, data_sharding),
                               out_shardings=(state_sharding, replicated), donate_argnums=(1,)).lower(
                                   jax.ShapeDtypeStruct(rng.shape, rng.dtype), state,
                                   config.model.inputs_spec(batch_size=config.batch_size)).compile()
        memory = compiled.memory_analysis()
        fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
                  "generated_code_size_in_bytes", "host_argument_size_in_bytes", "host_output_size_in_bytes",
                  "host_temp_size_in_bytes", "host_alias_size_in_bytes")
        result["compiled_memory"] = {field: int(getattr(memory, field, 0)) for field in fields}
        result["compiled_peak_buffer_estimate_bytes"] = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                                                        + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
        result["compile_seconds"] = time.time() - started
        result["executed_training_steps"] = 0
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
