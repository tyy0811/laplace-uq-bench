"""Training infrastructure for regressor and DDPM models."""

import json
import os
import time
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


def save_checkpoint(model, optimizer, epoch, val_loss, path, scheduler=None):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


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


def _find_latest_checkpoint(log_dir):
    """Find the latest epoch_N.pt checkpoint for warm-start resume."""
    import re
    best_epoch = -1
    best_path = None
    for p in log_dir.glob("epoch_*.pt"):
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = p
    return best_epoch, best_path


def _training_loop(model, train_loader, val_loader, optimizer, scheduler,
                   config, device, train_fn, val_fn, commit_fn=None):
    """Generic epoch loop shared by regressor and DDPM training.

    Args:
        commit_fn: Optional callable invoked after each checkpoint save
                   (e.g. Modal volume.commit for warm-start durability).
    """
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = config["training"]["epochs"]
    start_epoch = 0
    best_val = float("inf")
    history = []

    # Resume from latest periodic checkpoint if available
    history_path = log_dir / "history.json"
    last_epoch, ckpt_path = _find_latest_checkpoint(log_dir)
    if ckpt_path is not None and last_epoch < total_epochs:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["val_loss"]
        # Restore scheduler state directly (avoids LR corruption from replaying steps)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif scheduler:
            # Legacy checkpoint without scheduler state — replay as fallback
            for _ in range(start_epoch):
                scheduler.step()
        # Reload history so far
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        # Best checkpoint may have a better val_loss than the periodic one
        best_pt = log_dir / "best.pt"
        if best_pt.exists():
            best_ckpt = torch.load(best_pt, map_location=device)
            best_val = best_ckpt["val_loss"]
        print(f"Resuming from epoch {start_epoch} (best_val={best_val:.6f})")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_loss = val_fn(model, val_loader, device)
        if scheduler:
            scheduler.step()
        epoch_time = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e} | {epoch_time:.1f}s")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "epoch_time_seconds": epoch_time})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, log_dir / "best.pt", scheduler)

        save_every = config["logging"].get("save_every", 10)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, log_dir / f"epoch_{epoch+1}.pt", scheduler)
            # Persist history incrementally so resume knows prior epochs
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            if commit_fn:
                commit_fn()

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def train(config, device="cpu", commit_fn=None):
    """Full regressor training loop from config dict."""
    train_loader, val_loader = _make_loaders(config)

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    return _training_loop(model, train_loader, val_loader, optimizer, scheduler,
                          config, device, train_one_epoch, validate,
                          commit_fn=commit_fn)


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


def _is_improved_ddpm(ddpm_cfg):
    """Check whether config uses ImprovedDDPM features."""
    return any(k in ddpm_cfg for k in ("schedule", "prediction", "min_snr_gamma"))


def _build_ddpm(model, ddpm_cfg):
    """Instantiate DDPM or ImprovedDDPM from config dict."""
    if _is_improved_ddpm(ddpm_cfg):
        from .ddpm import ImprovedDDPM
        return ImprovedDDPM(
            model,
            T=ddpm_cfg["T"],
            beta_start=ddpm_cfg.get("beta_start", 1e-4),
            beta_end=ddpm_cfg.get("beta_end", 0.02),
            schedule=ddpm_cfg.get("schedule", "cosine"),
            prediction=ddpm_cfg.get("prediction", "v"),
            min_snr_gamma=ddpm_cfg.get("min_snr_gamma", 5.0),
        )
    from .ddpm import DDPM
    return DDPM(
        model,
        T=ddpm_cfg["T"],
        beta_start=ddpm_cfg["beta_start"],
        beta_end=ddpm_cfg["beta_end"],
    )


def train_ddpm(config, device="cpu", commit_fn=None):
    """Full DDPM or PhysicsDDPM training loop from config dict."""
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
        ddpm = _build_ddpm(model, ddpm_cfg).to(device)
        train_fn = train_ddpm_one_epoch
        val_fn = validate_ddpm

    optimizer = build_optimizer(ddpm, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    return _training_loop(ddpm, train_loader, val_loader, optimizer, scheduler,
                          config, device, train_fn, val_fn,
                          commit_fn=commit_fn)


def train_unconditional_ddpm_one_epoch(ddpm, loader, optimizer, device):
    """Train unconditional DDPM for one epoch (targets only, no conditioning)."""
    ddpm.train()
    total_loss = 0.0
    n_batches = 0
    for _cond, target in loader:
        target = target.to(device)
        loss = ddpm.training_step(target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate_unconditional_ddpm(ddpm, loader, device):
    """Compute average unconditional DDPM loss on validation set."""
    ddpm.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for _cond, target in loader:
            target = target.to(device)
            loss = ddpm.training_step(target)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def train_unconditional_ddpm(config, device="cpu", commit_fn=None):
    """Full unconditional DDPM training loop from config dict."""
    from .unconditional_ddpm import UnconditionalDDPM

    train_loader, val_loader = _make_loaders(config)

    model = build_model(config["model"]).to(device)
    ddpm_cfg = config["ddpm"]
    ddpm = UnconditionalDDPM(
        model,
        T=ddpm_cfg["T"],
        beta_start=ddpm_cfg.get("beta_start", 1e-4),
        beta_end=ddpm_cfg.get("beta_end", 0.02),
        schedule=ddpm_cfg.get("schedule", "cosine"),
        prediction=ddpm_cfg.get("prediction", "v"),
        min_snr_gamma=ddpm_cfg.get("min_snr_gamma", 5.0),
    ).to(device)

    optimizer = build_optimizer(ddpm, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    return _training_loop(ddpm, train_loader, val_loader, optimizer, scheduler,
                          config, device,
                          train_unconditional_ddpm_one_epoch,
                          validate_unconditional_ddpm,
                          commit_fn=commit_fn)


def train_cfm_one_epoch(cfm, loader, optimizer, device):
    """Train ConditionalFlowMatcher for one epoch. Returns average loss."""
    cfm.train()
    total_loss = 0.0
    n_batches = 0
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        loss = cfm.training_step(cond, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate_cfm(cfm, loader, device):
    """Compute average flow matching loss on validation set."""
    cfm.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            loss = cfm.training_step(cond, target)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def train_cfm(config, device="cpu", commit_fn=None):
    """Full Conditional Flow Matching training loop from config dict."""
    from .flow_matching import ConditionalFlowMatcher

    train_loader, val_loader = _make_loaders(config)

    model = build_model(config["model"]).to(device)
    fm_cfg = config["flow_matching"]
    cfm = ConditionalFlowMatcher(
        model,
        use_ot=fm_cfg.get("use_ot", True),
        n_sample_steps=fm_cfg.get("n_sample_steps", 50),
    ).to(device)

    optimizer = build_optimizer(cfm, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    return _training_loop(cfm, train_loader, val_loader, optimizer, scheduler,
                          config, device, train_cfm_one_epoch, validate_cfm,
                          commit_fn=commit_fn)


def train_ensemble(config, device="cpu", commit_fn=None):
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

        train(member_config, device=device, commit_fn=commit_fn)
