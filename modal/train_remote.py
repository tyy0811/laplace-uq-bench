"""Remote training on Modal A100 GPUs.

Usage:
    modal run modal/train_remote.py --config configs/unet_regressor.yaml
    modal run modal/train_remote.py --config configs/ddpm.yaml
    modal run modal/train_remote.py --config configs/ensemble_phase2.yaml
"""

import modal

app = modal.App("diffphys-training")

volume = modal.Volume.from_name("diffphys-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal/requirements.txt")
    .copy_local_dir("src", "/root/src")
    .copy_local_dir("configs", "/root/configs")
    .copy_local_file("pyproject.toml", "/root/pyproject.toml")
    .run_commands("cd /root && pip install -e .")
)


def _rewrite_paths(config):
    """Rewrite data/ and experiments/ paths to Modal volume mount."""
    for key in ("train", "val"):
        if key in config.get("data", {}):
            config["data"][key] = config["data"][key].replace("data/", "/data/")
    config["logging"]["log_dir"] = config["logging"]["log_dir"].replace(
        "experiments/", "/data/experiments/"
    )
    return config


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def train_model(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train, train_ddpm

    config = _rewrite_paths(load_config(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "ddpm" in config:
        train_ddpm(config, device=device)
    else:
        train(config, device=device)

    volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 6,
    volumes={"/data": volume},
)
def train_ensemble_remote(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train_ensemble

    config = _rewrite_paths(load_config(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ensemble(config, device=device)

    volume.commit()


@app.local_entrypoint()
def main(config: str):
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if "ensemble" in cfg:
        train_ensemble_remote.remote(config)
    else:
        train_model.remote(config)
