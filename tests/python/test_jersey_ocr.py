"""Tests for Phase 16.3 — Jersey OCR model.

Covers:
  - JerseyOCRNet architecture (input/output shape, forward pass)
  - JerseyOCR constructor raises FileNotFoundError for missing model
  - Identity locking logic (LOCK_READS consecutive identical reads → locked)
  - reset() / reset_all() clear state
  - check_quality_gate logic
  - Synthetic data generator: team colours registry, _render_base output size,
    _augment variant count, _augment determinism, labels.csv round-trip
  - Training utilities: JerseyDataset missing CSV, _verify_data, _build_loaders,
    _parse_args defaults and overrides
  - main() dry-run mode (no real training)

All tests run without GPU or a real trained model file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from models.cv.jersey_ocr import (
    INPUT_SIZE,
    JerseyOCR,
    JerseyOCRNet,
    LOCK_READS,
    NUM_CLASSES,
    TOP1_GATE,
    UNKNOWN_NUM,
    check_quality_gate,
)


# ===========================================================================
# JerseyOCRNet — architecture smoke tests
# ===========================================================================

class TestJerseyOCRNet:
    def test_output_shape(self):
        model = JerseyOCRNet()
        model.eval()
        x = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_batch_output_shape(self):
        model = JerseyOCRNet()
        model.eval()
        x = torch.zeros(8, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, NUM_CLASSES)

    def test_num_classes_is_100(self):
        assert NUM_CLASSES == 100

    def test_input_size_is_48(self):
        assert INPUT_SIZE == 48

    def test_forward_no_nan(self):
        model = JerseyOCRNet()
        model.eval()
        x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()


# ===========================================================================
# JerseyOCR constructor
# ===========================================================================

class TestJerseyOCRInit:
    def test_raises_if_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="(?i)jersey ocr model not found"):
            JerseyOCR(model_path=tmp_path / "nonexistent.pt")

    def test_error_mentions_train_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train-jersey-ocr"):
            JerseyOCR(model_path=tmp_path / "nonexistent.pt")


# ===========================================================================
# Identity locking
# ===========================================================================

def _make_ocr_from_net(net: JerseyOCRNet, tmp_path: Path, conf_thresh: float = 0.0) -> JerseyOCR:
    """Save net weights to a temp file and return a loaded JerseyOCR."""
    pt_path = tmp_path / "jersey_ocr.pt"
    torch.save(net.state_dict(), str(pt_path))
    ocr = JerseyOCR.__new__(JerseyOCR)
    ocr._device     = "cpu"
    ocr._conf_thresh = conf_thresh
    ocr._lock_reads  = LOCK_READS
    ocr._net         = net
    ocr._net.eval()
    from collections import defaultdict
    ocr._history = defaultdict(list)
    ocr._locked  = {}
    return ocr


class TestIdentityLocking:
    def _ocr_always_returns(self, number: int, conf: float = 0.9) -> JerseyOCR:
        """Build a JerseyOCR whose read() always returns (number, conf)."""
        net = JerseyOCRNet()
        ocr = _make_ocr_from_net(net, Path("/tmp"))
        ocr.read = lambda crop: (number, conf)  # type: ignore[method-assign]
        return ocr

    def test_not_locked_after_one_read(self):
        ocr = self._ocr_always_returns(42)
        _, _, locked = ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert not locked

    def test_not_locked_after_two_reads(self):
        ocr = self._ocr_always_returns(42)
        for _ in range(2):
            _, _, locked = ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert not locked

    def test_locked_after_lock_reads_consistent(self):
        ocr = self._ocr_always_returns(42)
        result = None
        for _ in range(LOCK_READS):
            result = ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert result is not None
        _, _, locked = result
        assert locked

    def test_locked_number_is_correct(self):
        ocr = self._ocr_always_returns(17)
        for _ in range(LOCK_READS):
            num, _, _ = ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert num == 17

    def test_inconsistent_reads_do_not_lock(self):
        net = JerseyOCRNet()
        ocr = _make_ocr_from_net(net, Path("/tmp"))
        # Alternate numbers
        call_count = [0]
        def alternating(crop):
            call_count[0] += 1
            return (42 if call_count[0] % 2 == 0 else 17, 0.9)
        ocr.read = alternating  # type: ignore[method-assign]
        for _ in range(LOCK_READS * 2):
            _, _, locked = ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert not locked

    def test_skip_inference_after_lock(self):
        ocr = self._ocr_always_returns(99)
        read_called = [0]
        orig_read = ocr.read
        def counting_read(crop):
            read_called[0] += 1
            return orig_read(crop)
        ocr.read = counting_read  # type: ignore[method-assign]
        for _ in range(LOCK_READS):
            ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        n_before = read_called[0]
        # 5 more calls after lock — should not increase read_called
        for _ in range(5):
            ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert read_called[0] == n_before

    def test_reset_clears_lock(self):
        ocr = self._ocr_always_returns(42)
        for _ in range(LOCK_READS):
            ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert ocr.is_locked(1)
        ocr.reset(1)
        assert not ocr.is_locked(1)

    def test_reset_all_clears_all_tracks(self):
        ocr = self._ocr_always_returns(10)
        for tid in [1, 2, 3]:
            for _ in range(LOCK_READS):
                ocr.update(tid, np.zeros((48, 48, 3), dtype=np.uint8))
        ocr.reset_all()
        assert not any(ocr.is_locked(tid) for tid in [1, 2, 3])

    def test_unknown_reads_do_not_accumulate(self):
        """UNKNOWN_NUM reads should not contribute to locking."""
        net = JerseyOCRNet()
        ocr = _make_ocr_from_net(net, Path("/tmp"))
        ocr.read = lambda crop: (UNKNOWN_NUM, 0.1)  # type: ignore[method-assign]
        for _ in range(LOCK_READS * 3):
            _, _, locked = ocr.update(1, np.zeros((48, 48, 3), dtype=np.uint8))
        assert not locked

    def test_multiple_tracks_independent(self):
        ocr = self._ocr_always_returns(55)
        for tid in [10, 20]:
            for _ in range(LOCK_READS):
                ocr.update(tid, np.zeros((48, 48, 3), dtype=np.uint8))
        assert ocr.is_locked(10)
        assert ocr.is_locked(20)
        ocr.reset(10)
        assert not ocr.is_locked(10)
        assert ocr.is_locked(20)    # unaffected


# ===========================================================================
# Quality gate
# ===========================================================================

class TestQualityGate:
    def test_passes_above_threshold(self):
        passed, val = check_quality_gate({"top1_acc": 0.92})
        assert passed
        assert val == 0.92

    def test_fails_below_threshold(self):
        passed, val = check_quality_gate({"top1_acc": 0.70})
        assert not passed
        assert val == 0.70

    def test_exactly_at_threshold(self):
        passed, _ = check_quality_gate({"top1_acc": TOP1_GATE})
        assert passed

    def test_empty_metrics_returns_false(self):
        passed, val = check_quality_gate({})
        assert not passed
        assert val == 0.0

    def test_none_metrics_no_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("models.cv.jersey_ocr.METRICS_JSON", tmp_path / "missing.json")
        passed, val = check_quality_gate(None)
        assert not passed
        assert val == 0.0

    def test_gate_value_is_085(self):
        assert TOP1_GATE == 0.85

    def test_lock_reads_is_3(self):
        assert LOCK_READS == 3


# ===========================================================================
# Synthetic data generator
# ===========================================================================

class TestTeamColors:
    def test_32_teams(self):
        from scripts.generate_jersey_data import TEAM_COLORS, TEAM_CODES
        assert len(TEAM_COLORS) == 32
        assert len(TEAM_CODES) == 32

    def test_each_team_has_bg_fg(self):
        from scripts.generate_jersey_data import TEAM_COLORS
        for code, colors in TEAM_COLORS.items():
            assert "bg" in colors, f"{code} missing 'bg'"
            assert "fg" in colors, f"{code} missing 'fg'"
            assert len(colors["bg"]) == 3, f"{code} bg not RGB"
            assert len(colors["fg"]) == 3, f"{code} fg not RGB"

    def test_no_duplicate_codes(self):
        from scripts.generate_jersey_data import TEAM_CODES
        assert len(set(TEAM_CODES)) == len(TEAM_CODES)


class TestRenderBase:
    def test_output_size(self):
        from scripts.generate_jersey_data import _render_base
        img = _render_base(42, "BOS")
        assert img.size == (INPUT_SIZE, INPUT_SIZE)

    def test_output_is_rgb(self):
        from scripts.generate_jersey_data import _render_base
        img = _render_base(1, "MTL")
        assert img.mode == "RGB"

    def test_background_colour_applied(self):
        from scripts.generate_jersey_data import TEAM_COLORS, _render_base
        team = "LAK"  # black background
        img = _render_base(0, team, font_size=1)  # tiny font → mostly bg
        arr = np.array(img)
        # Corner pixel should be close to background colour
        bg  = TEAM_COLORS[team]["bg"]
        corner = arr[0, 0]
        # Allow some tolerance
        assert all(abs(int(corner[i]) - bg[i]) < 30 for i in range(3))


class TestAugment:
    def test_returns_16_variants(self):
        from scripts.generate_jersey_data import _augment, _render_base
        rng  = np.random.default_rng(0)
        base = _render_base(99, "TOR")
        augs = _augment(base, rng)
        assert len(augs) == 16

    def test_variant_indices_unique(self):
        from scripts.generate_jersey_data import _augment, _render_base
        rng  = np.random.default_rng(0)
        base = _render_base(7, "EDM")
        augs = _augment(base, rng)
        variants = [a.variant for a in augs]
        assert len(set(variants)) == len(variants)

    def test_base_is_variant_0(self):
        from scripts.generate_jersey_data import _augment, _render_base
        rng  = np.random.default_rng(0)
        base = _render_base(11, "PHI")
        augs = _augment(base, rng)
        assert augs[0].variant == 0

    def test_all_images_correct_size(self):
        from scripts.generate_jersey_data import _augment, _render_base
        rng  = np.random.default_rng(0)
        base = _render_base(33, "WSH")
        augs = _augment(base, rng)
        for aug in augs:
            assert aug.image.size == (INPUT_SIZE, INPUT_SIZE)


class TestGeneratePipeline:
    def test_creates_labels_csv(self, tmp_path):
        from scripts.generate_jersey_data import generate
        generate(tmp_path, count_target=200, seed=0, verbose=False)
        assert (tmp_path / "labels.csv").exists()

    def test_creates_train_val_dirs(self, tmp_path):
        from scripts.generate_jersey_data import generate
        generate(tmp_path, count_target=200, seed=0, verbose=False)
        assert (tmp_path / "train").is_dir()
        assert (tmp_path / "val").is_dir()

    def test_labels_have_correct_columns(self, tmp_path):
        from scripts.generate_jersey_data import generate
        generate(tmp_path, count_target=200, seed=0, verbose=False)
        with open(tmp_path / "labels.csv", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
        for col in ("filename", "split", "number", "team", "variant"):
            assert col in cols

    def test_numbers_in_range(self, tmp_path):
        from scripts.generate_jersey_data import generate
        generate(tmp_path, count_target=200, seed=0, verbose=False)
        with open(tmp_path / "labels.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                n = int(row["number"])
                assert 1 <= n <= 99

    def test_returns_positive_count(self, tmp_path):
        from scripts.generate_jersey_data import generate
        n = generate(tmp_path, count_target=200, seed=0, verbose=False)
        assert n > 0

    def test_idempotent_without_force(self, tmp_path, capsys):
        from scripts.generate_jersey_data import main
        main(["--out", str(tmp_path), "--count", "200", "--force"])
        first_mtime = (tmp_path / "labels.csv").stat().st_mtime
        # Calling again without --force should skip and not recreate
        main(["--out", str(tmp_path), "--count", "200"])
        second_mtime = (tmp_path / "labels.csv").stat().st_mtime
        assert second_mtime == first_mtime


# ===========================================================================
# Training utilities
# ===========================================================================

class TestVerifyData:
    def test_raises_if_csv_missing(self, tmp_path):
        from scripts.train_jersey_ocr import _verify_data
        with pytest.raises(FileNotFoundError, match="(?i)jersey training data not found"):
            _verify_data(tmp_path)

    def test_raises_if_csv_too_small(self, tmp_path):
        from scripts.train_jersey_ocr import _verify_data
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "split", "number", "team", "variant"])
            w.writerow(["img.jpg", "train", "42", "BOS", "0"])
        with pytest.raises(ValueError, match="too small"):
            _verify_data(tmp_path)

    def test_passes_with_valid_data(self, tmp_path):
        from scripts.generate_jersey_data import generate
        from scripts.train_jersey_ocr import _verify_data
        generate(tmp_path, count_target=200, seed=0, verbose=False)
        _verify_data(tmp_path)   # should not raise


class TestJerseyDataset:
    def _make_data(self, tmp_path: Path, n: int = 20) -> Path:
        from scripts.generate_jersey_data import generate
        generate(tmp_path, count_target=n * 16, seed=0, verbose=False)
        return tmp_path

    def test_raises_if_csv_missing(self, tmp_path):
        from scripts.train_jersey_ocr import JerseyDataset
        with pytest.raises(FileNotFoundError):
            JerseyDataset(tmp_path, split="train")

    def test_len_positive(self, tmp_path):
        from scripts.train_jersey_ocr import JerseyDataset
        self._make_data(tmp_path, n=10)
        ds = JerseyDataset(tmp_path, split="train")
        assert len(ds) > 0

    def test_item_shape_and_label(self, tmp_path):
        from scripts.train_jersey_ocr import JerseyDataset
        self._make_data(tmp_path, n=10)
        ds = JerseyDataset(tmp_path, split="train")
        tensor, label = ds[0]
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert 1 <= label <= 99


class TestParseArgs:
    def test_defaults(self):
        from scripts.train_jersey_ocr import _parse_args, DEFAULT_EPOCHS, DEFAULT_BATCH, DEFAULT_LR
        args = _parse_args([])
        assert args.epochs == DEFAULT_EPOCHS
        assert args.batch  == DEFAULT_BATCH
        assert abs(args.lr - DEFAULT_LR) < 1e-9
        assert not args.dry_run
        assert not args.force
        assert not args.no_gen

    def test_overrides(self):
        from scripts.train_jersey_ocr import _parse_args
        args = _parse_args(["--epochs", "5", "--batch", "32", "--dry-run", "--force"])
        assert args.epochs == 5
        assert args.batch  == 32
        assert args.dry_run
        assert args.force


class TestDryRun:
    def test_dry_run_no_training(self, tmp_path):
        from scripts.train_jersey_ocr import main
        from scripts.generate_jersey_data import generate
        data_dir = tmp_path / "data"
        generate(data_dir, count_target=200, seed=0, verbose=False)
        model_pt = tmp_path / "model.pt"
        # dry-run should not create a model file
        main(["--data", str(data_dir), "--dry-run", "--no-gen"])
        assert not model_pt.exists()
