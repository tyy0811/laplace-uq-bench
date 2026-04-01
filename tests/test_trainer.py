"""Tests for training infrastructure."""

import numpy as np
import torch
import pytest
import yaml
from pathlib import Path

from diffphys.model.trainer import (
    build_model,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    train_one_epoch,
    validate,
    load_config,
    train_ddpm_one_epoch,
    validate_ddpm,
    train_ddpm,
    train_ensemble,
)
from diffphys.model.unet import ConditionalUNet
from diffphys.model.ddpm import DDPM
from diffphys.data.dataset import LaplacePDEDataset


@pytest.fixture
def tiny_npz(tmp_path):
    n, nx = 16, 16
    rng = np.random.default_rng(42)
    for split in ("train", "val"):
        np.savez(
            tmp_path / f"{split}.npz",
            fields=rng.standard_normal((n, nx, nx)).astype(np.float32),
            bc_top=rng.standard_normal((n, nx)).astype(np.float32),
            bc_bottom=rng.standard_normal((n, nx)).astype(np.float32),
            bc_left=rng.standard_normal((n, nx)).astype(np.float32),
            bc_right=rng.standard_normal((n, nx)).astype(np.float32),
        )
    return tmp_path


@pytest.fixture
def config(tiny_npz):
    return {
        "model": {"type": "unet", "in_channels": 8, "out_channels": 1, "base_channels": 16, "channel_mult": [1, 2, 4]},
        "training": {"batch_size": 4, "lr": 1e-3, "epochs": 2, "weight_decay": 1e-4, "scheduler": "cosine"},
        "data": {"train": str(tiny_npz / "train.npz"), "val": str(tiny_npz / "val.npz")},
        "logging": {"log_dir": str(tiny_npz / "logs"), "save_every": 1},
    }


class TestBuildModel:
    def test_builds_unet(self, config):
        model = build_model(config["model"])
        assert isinstance(model, ConditionalUNet)

    def test_respects_channel_config(self, config):
        config["model"]["base_channels"] = 32
        model = build_model(config["model"])
        # First conv should have 32 output channels
        assert model.enc_in.out_channels == 32


class TestTrainOneEpoch:
    def test_returns_finite_loss(self, config, tiny_npz):
        model = build_model(config["model"])
        optimizer = build_optimizer(model, config["training"])
        ds = LaplacePDEDataset(config["data"]["train"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        avg_loss = train_one_epoch(model, loader, optimizer, "cpu")
        assert np.isfinite(avg_loss)
        assert avg_loss > 0

    def test_loss_decreases_over_epochs(self, config, tiny_npz):
        model = build_model(config["model"])
        optimizer = build_optimizer(model, config["training"])
        ds = LaplacePDEDataset(config["data"]["train"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        loss1 = train_one_epoch(model, loader, optimizer, "cpu")
        loss2 = train_one_epoch(model, loader, optimizer, "cpu")
        loss3 = train_one_epoch(model, loader, optimizer, "cpu")
        # After a few epochs on tiny data, loss should decrease
        assert loss3 < loss1


class TestValidate:
    def test_returns_finite_loss(self, config, tiny_npz):
        model = build_model(config["model"])
        ds = LaplacePDEDataset(config["data"]["val"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        val_loss = validate(model, loader, "cpu")
        assert np.isfinite(val_loss)


class TestTrainDDPM:
    def test_ddpm_one_epoch_returns_finite_loss(self, config, tiny_npz):
        config["model"]["in_channels"] = 9
        config["model"]["time_emb_dim"] = 32
        model = build_model(config["model"])
        ddpm = DDPM(model, T=5)
        optimizer = build_optimizer(ddpm, config["training"])
        ds = LaplacePDEDataset(config["data"]["train"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        avg_loss = train_ddpm_one_epoch(ddpm, loader, optimizer, "cpu")
        assert np.isfinite(avg_loss)
        assert avg_loss > 0

    def test_train_ddpm_full_loop(self, config, tiny_npz):
        config["model"]["in_channels"] = 9
        config["model"]["time_emb_dim"] = 32
        config["ddpm"] = {"T": 5, "beta_start": 1e-4, "beta_end": 0.02}
        config["training"]["epochs"] = 1
        ddpm, history = train_ddpm(config, device="cpu")
        assert len(history) == 1
        assert np.isfinite(history[0]["train_loss"])

    def test_train_physics_ddpm_full_loop(self, config, tiny_npz):
        config["model"]["in_channels"] = 9
        config["model"]["time_emb_dim"] = 32
        config["ddpm"] = {"T": 5, "beta_start": 1e-4, "beta_end": 0.02}
        config["physics"] = {"residual_weight": 0.1}
        config["training"]["epochs"] = 1
        phys_ddpm, history = train_ddpm(config, device="cpu")
        assert len(history) == 1
        assert np.isfinite(history[0]["train_loss"])
        from diffphys.model.physics_ddpm import PhysicsDDPM
        assert isinstance(phys_ddpm, PhysicsDDPM)


class TestTrainEnsemble:
    def test_trains_multiple_members(self, config, tiny_npz):
        config["ensemble"] = {"n_members": 2, "seeds": [0, 1]}
        config["training"]["epochs"] = 1
        train_ensemble(config, device="cpu")
        # Should create checkpoints for each member
        assert (Path(config["logging"]["log_dir"]) / "member_0" / "best.pt").exists()
        assert (Path(config["logging"]["log_dir"]) / "member_1" / "best.pt").exists()

    def test_n_members_seeds_mismatch_raises(self, config, tiny_npz):
        config["ensemble"] = {"n_members": 5, "seeds": [0, 1]}
        with pytest.raises(ValueError, match="n_members=5 but len\\(seeds\\)=2"):
            train_ensemble(config, device="cpu")


class TestSchedulerResume:
    def test_lr_continuity_across_save_restore(self, config):
        """Restored scheduler must produce the same LR as uninterrupted training."""
        model = build_model(config["model"])
        optimizer = build_optimizer(model, config["training"])
        scheduler = build_scheduler(optimizer, config["training"])

        # Run 1 epoch, save checkpoint with scheduler
        scheduler.step()
        lr_after_epoch1 = optimizer.param_groups[0]["lr"]

        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ckpt.pt")
            save_checkpoint(model, optimizer, 0, 1.0, path, scheduler)

            # Create fresh optimizer+scheduler and restore
            optimizer2 = build_optimizer(model, config["training"])
            scheduler2 = build_scheduler(optimizer2, config["training"])
            ckpt = torch.load(path, map_location="cpu")
            optimizer2.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler2.load_state_dict(ckpt["scheduler_state_dict"])

            # Next step should match what uninterrupted training would give
            scheduler.step()
            scheduler2.step()
            lr_uninterrupted = optimizer.param_groups[0]["lr"]
            lr_restored = optimizer2.param_groups[0]["lr"]
            assert lr_restored == pytest.approx(lr_uninterrupted, rel=1e-6), \
                f"LR diverged: restored={lr_restored}, expected={lr_uninterrupted}"


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path, config):
        path = tmp_path / "test_config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        loaded = load_config(str(path))
        assert loaded["model"]["type"] == "unet"
