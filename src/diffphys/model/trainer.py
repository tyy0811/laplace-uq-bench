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


def train(config, device="cpu"):
    """Full training loop from config dict."""
    from ..data.dataset import LaplacePDEDataset

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    train_ds = LaplacePDEDataset(config["data"]["train"])
    val_ds = LaplacePDEDataset(config["data"]["val"])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        num_workers=0,
    )

    best_val = float("inf")
    history = []

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
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
