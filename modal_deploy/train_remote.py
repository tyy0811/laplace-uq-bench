"""Remote training on Modal T4 GPUs.

Usage:
    modal run modal_deploy/train_remote.py --config configs/ddpm_improved.yaml
    modal run modal_deploy/train_remote.py --config configs/flow_matching.yaml
    modal run modal_deploy/train_remote.py --config configs/ensemble_phase2.yaml
    modal run modal_deploy/train_remote.py --config configs/flow_matching.yaml --config2 configs/ddpm_improved.yaml
"""

import modal

app = modal.App("diffphys-training")

volume = modal.Volume.from_name("diffphys-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
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
    gpu="T4",
    timeout=3600 * 12,
    volumes={"/data": volume},
)
def train_model(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train, train_ddpm, train_cfm, train_unconditional_ddpm

    config = _rewrite_paths(load_config(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit = volume.commit

    if config.get("unconditional"):
        train_unconditional_ddpm(config, device=device, commit_fn=commit)
    elif "ddpm" in config:
        train_ddpm(config, device=device, commit_fn=commit)
    elif "flow_matching" in config:
        train_cfm(config, device=device, commit_fn=commit)
    else:
        train(config, device=device, commit_fn=commit)

    volume.commit()


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def train_ensemble_remote(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train_ensemble

    config = _rewrite_paths(load_config(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ensemble(config, device=device, commit_fn=volume.commit)

    volume.commit()


@app.local_entrypoint()
def main(config: str, config2: str = None):
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    handles = []
    if "ensemble" in cfg:
        handles.append(train_ensemble_remote.spawn(config))
    else:
        handles.append(train_model.spawn(config))

    if config2:
        with open(config2) as f:
            cfg2 = yaml.safe_load(f)
        if "ensemble" in cfg2:
            handles.append(train_ensemble_remote.spawn(config2))
        else:
            handles.append(train_model.spawn(config2))

    for h in handles:
        h.get()
    print("All training runs complete.")
