"""Check AtomBridge dependencies without downloading models or starting training."""

import importlib
import importlib.metadata
import json
import pathlib
import sys


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts" / "autobio_scripts"))
    results = {}

    def check(name, function):
        try:
            detail = function()
            results[name] = {"ok": True, "detail": str(detail)}
            print(f"PASS {name}: {detail}", flush=True)
        except Exception as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)

    modules = {
        "mujoco": "mujoco",
        "toppra": "toppra",
        "zstandard": "zstandard",
        "trimesh": "trimesh",
        "shapely": "shapely",
        "triangle": "triangle",
        "manifold3d": "manifold3d",
        "usd-core": "pxr.Usd",
        "torch": "torch",
        "torchvision": "torchvision",
        "jax": "jax",
        "flax": "flax",
        "lerobot": "lerobot.datasets.lerobot_dataset",
        "transformers": "transformers",
        "openai": "openai",
    }
    for distribution, module in modules.items():
        def load(distribution=distribution, module=module):
            imported = importlib.import_module(module)
            return f"{importlib.metadata.version(distribution)} ({imported.__file__})"

        check(distribution, load)

    def clip_classes():
        from transformers import CLIPModel, CLIPProcessor
        return f"{CLIPModel.__name__}, {CLIPProcessor.__name__}"

    check("CLIP classes", clip_classes)

    def openpi_config():
        from openpi.training.config import get_config
        config = get_config("mani_thermalcycler_pi05")
        return f"{config.name}, steps={config.num_train_steps}, batch={config.batch_size}"

    check("OpenPI config", openpi_config)

    def simulation_modules():
        import thermal_cycler_tasks
        import centrifuge5910_tasks
        return f"{len(thermal_cycler_tasks.DEFAULT_DATASET_TASKS)} thermal + {len(centrifuge5910_tasks.DEFAULT_DATASET_TASKS)} centrifuge tasks"

    check("expert modules and native plugin", simulation_modules)

    for scene in ("thermal_cycler_tasks", "centrifuge_5910_tasks"):
        def compile_scene(scene=scene):
            import mujoco
            model = mujoco.MjModel.from_xml_path(str(root / "model" / "scene" / f"{scene}.xml"))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            return f"nq={model.nq}, nu={model.nu}, cameras={model.ncam}"

        check(f"scene {scene}", compile_scene)

    def toppra_trajectory():
        import numpy as np
        from kinematics import Pose
        from topp import Topp
        planner = Topp(2, 1.5, 1.0, lambda position, quaternion: position[:2])
        poses = [Pose(np.array([t, t / 2, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])) for t in (0.0, 0.5, 1.0)]
        trajectory = planner.jnt_traj(poses)
        if trajectory is None or not np.isfinite(trajectory.duration) or trajectory.duration <= 0:
            raise RuntimeError("TOPPRA did not produce a finite trajectory")
        samples = trajectory.eval(np.linspace(0, trajectory.duration, 10))
        if samples.shape != (10, 2) or not np.isfinite(samples).all():
            raise RuntimeError("TOPPRA returned invalid samples")
        return f"duration={trajectory.duration:.3f}s, samples={samples.shape}"

    check("TOPPRA trajectory", toppra_trajectory)
    output = root / "artifacts" / "environment" / "setup_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return 0 if all(result["ok"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
