"""Training infrastructure for regressor and DDPM models."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .unet import ConditionalUNet


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    """Construct model from config dict."""
    mtype = model_cfg["type"]
    if mtype == "unet":
        return ConditionalUNet(
            in_ch=model_cfg.get("in_channels", 8),
            out_ch=model_cfg.get("out_channels", 1),
            base_ch=model_cfg.get("base_channels", 64),
            ch_mult=tuple(model_cfg.get("channel_mult", [1, 2, 4])),
            time_emb_dim=model_cfg.get("time_emb_dim", None),
        )
    elif mtype == "fno":
        from .fno import FNO2d
        return FNO2d(
            in_ch=model_cfg.get("in_channels", 8),
            out_ch=model_cfg.get("out_channels", 1),
            width=model_cfg.get("width", 40),
            modes=model_cfg.get("modes", 12),
            n_layers=model_cfg.get("n_layers", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def build_optimizer(model, train_cfg):
    return torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0),
    )


def build_scheduler(optimizer, train_cfg):
    if train_cfg.get("scheduler") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    return None


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        pred = model(cond)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate(model, loader, device):
    """Compute average MSE on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            pred = model(cond)
            total_loss += F.mse_loss(pred, target).item()
            n_batches += 1
    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def _make_loaders(config):
    """Build train/val DataLoaders from config, respecting regime.

    Validation always uses "exact" regime for deterministic loss
    comparison and stable best-checkpoint selection, even when
    training uses "mixed" augmentation.
    """
    from ..data.dataset import LaplacePDEDataset

    regime = config.get("training", {}).get("regime", "exact")
    train_ds = LaplacePDEDataset(config["data"]["train"], regime=regime)
    val_ds = LaplacePDEDataset(config["data"]["val"], regime="exact")
    bs = config["training"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, num_workers=0,
    )
    return train_loader, val_loader


def _training_loop(model, train_loader, val_loader, optimizer, scheduler,
                   config, device, train_fn, val_fn):
    """Generic epoch loop shared by regressor and DDPM training."""
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_loss = val_fn(model, val_loader, device)
        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, log_dir / "best.pt")

        if (epoch + 1) % config["logging"].get("save_every", 10) == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, log_dir / f"epoch_{epoch+1}.pt")

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def train(config, device="cpu"):
    """Full regressor training loop from config dict."""
    train_loader, val_loader = _make_loaders(config)

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    return _training_loop(model, train_loader, val_loader, optimizer, scheduler,
                          config, device, train_one_epoch, validate)


def train_ddpm_one_epoch(ddpm, loader, optimizer, device):
    """Train DDPM for one epoch. Returns average loss."""
    ddpm.train()
    total_loss = 0.0
    n_batches = 0
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        loss = ddpm.training_step(cond, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate_ddpm(ddpm, loader, device):
    """Compute average DDPM loss on validation set."""
    ddpm.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            loss = ddpm.training_step(cond, target)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def train_physics_ddpm_one_epoch(ddpm, loader, optimizer, device):
    """Train PhysicsDDPM for one epoch. Returns average total loss."""
    ddpm.train()
    total_loss = 0.0
    n_batches = 0
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        losses = ddpm.training_step(cond, target)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        total_loss += losses["total"].item()
        n_batches += 1
    return total_loss / n_batches


def validate_physics_ddpm(ddpm, loader, device):
    """Compute average PhysicsDDPM total loss on validation set."""
    ddpm.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            losses = ddpm.training_step(cond, target)
            total_loss += losses["total"].item()
            n_batches += 1
    return total_loss / n_batches


def train_ddpm(config, device="cpu"):
    """Full DDPM or PhysicsDDPM training loop from config dict."""
    from .ddpm import DDPM

    train_loader, val_loader = _make_loaders(config)

    model = build_model(config["model"]).to(device)
    ddpm_cfg = config["ddpm"]

    if "physics" in config:
        from .physics_ddpm import PhysicsDDPM
        ddpm = PhysicsDDPM(
            model, T=ddpm_cfg["T"],
            residual_weight=config["physics"]["residual_weight"],
            beta_start=ddpm_cfg["beta_start"],
            beta_end=ddpm_cfg["beta_end"],
        ).to(device)
        train_fn = train_physics_ddpm_one_epoch
        val_fn = validate_physics_ddpm
    else:
        ddpm = DDPM(model, T=ddpm_cfg["T"],
                    beta_start=ddpm_cfg["beta_start"],
                    beta_end=ddpm_cfg["beta_end"]).to(device)
        train_fn = train_ddpm_one_epoch
        val_fn = validate_ddpm

    optimizer = build_optimizer(ddpm, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    return _training_loop(ddpm, train_loader, val_loader, optimizer, scheduler,
                          config, device, train_fn, val_fn)


def train_ensemble(config, device="cpu"):
    """Train K independent U-Nets with different seeds."""
    ens_cfg = config["ensemble"]
    seeds = ens_cfg["seeds"]
    n_members = ens_cfg["n_members"]
    if len(seeds) != n_members:
        raise ValueError(
            f"ensemble.n_members={n_members} but len(seeds)={len(seeds)}; "
            f"these must match to ensure the correct experiment size"
        )
    base_log_dir = config["logging"]["log_dir"]

    for i, seed in enumerate(seeds):
        print(f"\n=== Training ensemble member {i+1}/{n_members} (seed={seed}) ===")
        torch.manual_seed(seed)

        member_config = {**config}
        member_config["logging"] = {
            **config["logging"],
            "log_dir": f"{base_log_dir}/member_{i}",
        }

        train(member_config, device=device)
