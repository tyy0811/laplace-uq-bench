"""Tests for conditional flow matching."""

import torch
import pytest
from diffphys.model.flow_matching import (
    FlowMatchingSchedule,
    OTCouplingMatcher,
    ConditionalFlowMatcher,
)
from diffphys.model.unet import ConditionalUNet


class TestFlowMatchingSchedule:
    def test_interpolate_at_t0(self):
        schedule = FlowMatchingSchedule()
        x_0 = torch.randn(4, 1, 8, 8)
        x_1 = torch.randn(4, 1, 8, 8)
        t = torch.zeros(4)
        x_t = schedule.interpolate(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_0)

    def test_interpolate_at_t1(self):
        schedule = FlowMatchingSchedule()
        x_0 = torch.randn(4, 1, 8, 8)
        x_1 = torch.randn(4, 1, 8, 8)
        t = torch.ones(4)
        x_t = schedule.interpolate(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_1)

    def test_interpolate_midpoint(self):
        schedule = FlowMatchingSchedule()
        x_0 = torch.zeros(4, 1, 8, 8)
        x_1 = torch.ones(4, 1, 8, 8)
        t = torch.full((4,), 0.5)
        x_t = schedule.interpolate(x_0, x_1, t)
        torch.testing.assert_close(x_t, 0.5 * torch.ones_like(x_t))

    def test_velocity_target_is_difference(self):
        schedule = FlowMatchingSchedule()
        x_0 = torch.randn(4, 1, 8, 8)
        x_1 = torch.randn(4, 1, 8, 8)
        target = schedule.compute_target(x_0, x_1)
        torch.testing.assert_close(target, x_1 - x_0)

    def test_sample_time_range(self):
        schedule = FlowMatchingSchedule()
        t = schedule.sample_time(10000, device=torch.device("cpu"))
        assert t.min() >= 0.0
        assert t.max() <= 1.0


class TestOTCouplingMatcher:
    def test_output_shape(self):
        matcher = OTCouplingMatcher()
        x_0 = torch.randn(8, 1, 8, 8)
        x_1 = torch.randn(8, 1, 8, 8)
        x_0_coupled = matcher.find_coupling(x_0, x_1)
        assert x_0_coupled.shape == x_0.shape

    def test_is_permutation(self):
        matcher = OTCouplingMatcher()
        x_0 = torch.randn(8, 1, 4, 4)
        x_1 = torch.randn(8, 1, 4, 4)
        x_0_coupled = matcher.find_coupling(x_0, x_1)
        for i in range(8):
            found = any(torch.allclose(x_0_coupled[i], x_0[j], atol=1e-6) for j in range(8))
            assert found, f"Row {i} not found in input"


class TestConditionalFlowMatcher:
    @pytest.fixture
    def cfm(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        return ConditionalFlowMatcher(model, use_ot=False, n_sample_steps=5)

    def test_training_step_returns_loss(self, cfm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = cfm.training_step(cond, target)
        assert loss.shape == ()
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_training_step_gradient_flow(self, cfm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = cfm.training_step(cond, target)
        loss.backward()
        for p in cfm.model.parameters():
            assert p.grad is not None

    def test_sample_shape(self, cfm):
        cond = torch.randn(2, 8, 16, 16)
        samples = cfm.sample(cond, n_samples=3)
        assert samples.shape == (3, 2, 1, 16, 16)

    def test_sample_finite(self, cfm):
        cond = torch.randn(1, 8, 16, 16)
        samples = cfm.sample(cond, n_samples=2)
        assert torch.isfinite(samples).all()

    def test_sample_different_per_call(self, cfm):
        cond = torch.randn(1, 8, 16, 16)
        s1 = cfm.sample(cond, n_samples=1)
        s2 = cfm.sample(cond, n_samples=1)
        assert not torch.allclose(s1, s2)

    def test_ot_coupling_training(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        cfm_ot = ConditionalFlowMatcher(model, use_ot=True, n_sample_steps=5)
        cond = torch.randn(4, 8, 16, 16)
        target = torch.randn(4, 1, 16, 16)
        loss = cfm_ot.training_step(cond, target)
        assert torch.isfinite(loss)
