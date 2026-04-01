"""Tests for partial checkpoint resume validation logic."""
import json
import os
import sys
from unittest.mock import MagicMock

import pytest

# evaluate_remote.py imports modal at top level, but our helpers are pure Python.
# Mock modal so we can import without the dependency installed.
sys.modules.setdefault("modal", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modal_deploy"))
from evaluate_remote import _load_partial_with_validation, _save_partial


RUN_PARAMS = {"max_samples": 300, "dataset": "test_ood.npz", "regime": "exact"}


class TestLoadPartialWithValidation:
    def test_no_partial_returns_empty(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        results, total_time = _load_partial_with_validation(path, RUN_PARAMS)
        assert results == {}
        assert total_time == 0.0

    def test_matching_params_resumes(self, tmp_path):
        path = str(tmp_path / "partial.json")
        _save_partial(path, {"model_a": {"metric": 1.0}}, 42.0, RUN_PARAMS)
        results, total_time = _load_partial_with_validation(path, RUN_PARAMS)
        assert results == {"model_a": {"metric": 1.0}}
        assert total_time == 42.0

    def test_mismatched_params_raises(self, tmp_path):
        path = str(tmp_path / "partial.json")
        _save_partial(path, {"model_a": {"metric": 1.0}}, 10.0, RUN_PARAMS)
        different_params = {"max_samples": 100, "dataset": "test_ood.npz", "regime": "exact"}
        with pytest.raises(ValueError, match="Use --fresh"):
            _load_partial_with_validation(path, different_params)

    def test_legacy_partial_without_run_params_raises(self, tmp_path):
        path = str(tmp_path / "partial.json")
        # Simulate a pre-fix partial file with no run_params key
        with open(path, "w") as f:
            json.dump({"results": {"old_model": {}}, "eval_time_seconds": 5.0}, f)
        with pytest.raises(ValueError, match="Use --fresh"):
            _load_partial_with_validation(path, RUN_PARAMS)

    def test_legacy_partial_with_null_run_params_raises(self, tmp_path):
        path = str(tmp_path / "partial.json")
        with open(path, "w") as f:
            json.dump({"results": {}, "eval_time_seconds": 0.0, "run_params": None}, f)
        with pytest.raises(ValueError, match="Use --fresh"):
            _load_partial_with_validation(path, RUN_PARAMS)


class TestSavePartial:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "partial.json")
        _save_partial(path, {"m": {"x": 1}}, 3.14, RUN_PARAMS)
        with open(path) as f:
            data = json.load(f)
        assert data["results"] == {"m": {"x": 1}}
        assert data["eval_time_seconds"] == 3.14
        assert data["run_params"] == RUN_PARAMS
