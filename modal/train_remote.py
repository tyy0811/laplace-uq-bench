"""Remote training on Modal A100 GPUs.

Usage:
    modal run modal/train_remote.py --config configs/unet_regressor.yaml
    modal run modal/train_remote.py --config configs/ddpm.yaml
"""

import modal

app = modal.App("diffphys-training")

volume = modal.Volume.from_name("diffphys-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal/requirements.txt")
    .copy_local_dir("src", "/root/src")
    .copy_local_dir("configs", "/root/configs")
    .run_commands("cd /root && pip install -e .")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def train_model(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train, train_ddpm

    config = load_config(config_path)

    # Point data paths to the volume mount
    for key in ("train", "val"):
        if key in config.get("data", {}):
            original = config["data"][key]
            config["data"][key] = original.replace("data/", "/data/")

    config["logging"]["log_dir"] = config["logging"]["log_dir"].replace(
        "experiments/", "/data/experiments/"
    )

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
def train_ensemble(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for key in ("train", "val"):
        config["data"][key] = config["data"][key].replace("data/", "/data/")

    seeds = config["ensemble"]["seeds"]
    for i, seed in enumerate(seeds):
        print(f"\n=== Training ensemble member {i+1}/{len(seeds)} (seed={seed}) ===")
        torch.manual_seed(seed)

        member_config = {**config}
        member_config["logging"] = {
            **config["logging"],
            "log_dir": config["logging"]["log_dir"].replace(
                "experiments/", "/data/experiments/"
            ) + f"/member_{i}",
        }

        train(member_config, device=device)

    volume.commit()


@app.local_entrypoint()
def main(config: str):
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if "ensemble" in cfg:
        train_ensemble.remote(config)
    else:
        train_model.remote(config)
