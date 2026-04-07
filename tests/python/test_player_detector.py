"""Tests for Phase 16.2 — Player Detection Model.

Tests cover:
  - Detection dataclass properties (cx/cy/w/h/to_dict)
  - CLASS_NAMES registry (correct count and order)
  - PlayerDetector constructor raises FileNotFoundError for missing model
  - quality gate check_quality_gate logic
  - training script utilities: _verify_data, _prepare_split, _parse_args
  - _metrics_from_csv fallback parser

All tests run without a real model file or GPU.
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from models.cv.player_detector import (
    CLASS_NAMES,
    NUM_CLASSES,
    MAP50_GATE,
    Detection,
    check_quality_gate,
    load_metrics,
)


# ===========================================================================
# Detection dataclass
# ===========================================================================

class TestDetection:
    def _det(self, x1=100.0, y1=200.0, x2=180.0, y2=300.0, cls_id=0, conf=0.9):
        return Detection(cls_id, CLASS_NAMES[cls_id], conf, x1, y1, x2, y2)

    def test_cx(self):
        d = self._det(x1=100, x2=200, y1=0, y2=0)
        assert d.cx == 150.0

    def test_cy(self):
        d = self._det(x1=0, x2=0, y1=100, y2=300)
        assert d.cy == 200.0

    def test_w(self):
        d = self._det(x1=50, x2=150)
        assert d.w == 100.0

    def test_h(self):
        d = self._det(y1=10, y2=90)
        assert d.h == 80.0

    def test_to_dict_keys(self):
        d = self._det()
        result = d.to_dict()
        for key in ("class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "w", "h"):
            assert key in result

    def test_to_dict_class_name(self):
        d = Detection(5, "puck", 0.75, 400, 300, 420, 315)
        assert d.to_dict()["class_name"] == "puck"
        assert d.to_dict()["class_id"] == 5

    def test_to_dict_rounded(self):
        d = Detection(0, "player_home", 0.876543, 10.0, 20.0, 30.0, 40.0)
        r = d.to_dict()
        assert r["confidence"] == round(0.876543, 4)


# ===========================================================================
# CLASS_NAMES registry
# ===========================================================================

class TestClassRegistry:
    def test_count(self):
        assert NUM_CLASSES == 6
        assert len(CLASS_NAMES) == 6

    def test_order(self):
        assert CLASS_NAMES[0] == "player_home"
        assert CLASS_NAMES[1] == "player_away"
        assert CLASS_NAMES[2] == "goalie_home"
        assert CLASS_NAMES[3] == "goalie_away"
        assert CLASS_NAMES[4] == "referee"
        assert CLASS_NAMES[5] == "puck"

    def test_no_duplicates(self):
        assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)


# ===========================================================================
# PlayerDetector constructor
# ===========================================================================

class TestPlayerDetectorInit:
    def test_raises_if_model_missing(self, tmp_path):
        from models.cv.player_detector import PlayerDetector
        with pytest.raises(FileNotFoundError, match="(?i)player detector model not found"):
            PlayerDetector(model_path=tmp_path / "nonexistent.pt")

    def test_error_message_mentions_train_command(self, tmp_path):
        from models.cv.player_detector import PlayerDetector
        with pytest.raises(FileNotFoundError, match="train-cv-detector"):
            PlayerDetector(model_path=tmp_path / "nonexistent.pt")


# ===========================================================================
# Quality gate
# ===========================================================================

class TestQualityGate:
    def test_passes_above_threshold(self):
        passed, val = check_quality_gate({"map50": 0.85})
        assert passed
        assert val == 0.85

    def test_fails_below_threshold(self):
        passed, val = check_quality_gate({"map50": 0.75})
        assert not passed
        assert val == 0.75

    def test_exactly_at_threshold(self):
        passed, val = check_quality_gate({"map50": MAP50_GATE})
        assert passed

    def test_no_metrics_returns_false(self):
        passed, val = check_quality_gate(None)
        assert not passed
        assert val == 0.0

    def test_missing_map50_key(self):
        passed, val = check_quality_gate({"precision": 0.9})
        assert not passed
        assert val == 0.0

    def test_gate_value(self):
        assert MAP50_GATE == 0.82


# ===========================================================================
# Training utilities
# ===========================================================================

class TestVerifyData:
    def test_raises_if_img_dir_missing(self, tmp_path):
        from scripts.train_cv_detector import _verify_data
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            _verify_data(tmp_path / "missing_images", tmp_path / "labels")

    def test_raises_if_lbl_dir_missing(self, tmp_path):
        from scripts.train_cv_detector import _verify_data
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Label directory not found"):
            _verify_data(img_dir, tmp_path / "missing_labels")

    def test_raises_if_no_sequences(self, tmp_path):
        from scripts.train_cv_detector import _verify_data
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No sequence directories"):
            _verify_data(img_dir, lbl_dir)

    def test_raises_if_no_images(self, tmp_path):
        from scripts.train_cv_detector import _verify_data
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        (img_dir / "seq1").mkdir(parents=True)
        (lbl_dir / "seq1").mkdir(parents=True)
        with pytest.raises(ValueError, match="No images found"):
            _verify_data(img_dir, lbl_dir)


class TestPrepareSplit:
    def _make_dataset(self, tmp_path: Path, n_seqs: int = 5, n_frames: int = 4) -> tuple[Path, Path]:
        img_root = tmp_path / "aug_images"
        lbl_root = tmp_path / "aug_labels"
        for s in range(n_seqs):
            seq_name = f"seq{s:02d}"
            (img_root / seq_name).mkdir(parents=True)
            (lbl_root / seq_name).mkdir(parents=True)
            for f in range(n_frames):
                img = img_root / seq_name / f"{f:06d}_a0.jpg"
                lbl = lbl_root / seq_name / f"{f:06d}_a0.txt"
                img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)   # minimal JPEG header
                lbl.write_text(f"0 0.5 0.5 0.1 0.1\n")
        return img_root, lbl_root

    def test_creates_split_yaml(self, tmp_path):
        from scripts.train_cv_detector import _prepare_split
        img, lbl = self._make_dataset(tmp_path, n_seqs=5)
        yaml = _prepare_split(img, lbl, tmp_path / "split")
        assert yaml.exists()
        content = yaml.read_text()
        assert "nc: 6" in content
        assert "player_home" in content

    def test_creates_train_and_val_dirs(self, tmp_path):
        from scripts.train_cv_detector import _prepare_split
        img, lbl = self._make_dataset(tmp_path, n_seqs=5)
        split_dir = tmp_path / "split"
        _prepare_split(img, lbl, split_dir)
        assert (split_dir / "images" / "train").is_dir()
        assert (split_dir / "images" / "val").is_dir()
        assert (split_dir / "labels" / "train").is_dir()
        assert (split_dir / "labels" / "val").is_dir()

    def test_val_has_at_least_one_seq(self, tmp_path):
        from scripts.train_cv_detector import _prepare_split
        # Only 1 sequence — val still gets 1 (degenerate but handled)
        img, lbl = self._make_dataset(tmp_path, n_seqs=1)
        split_dir = tmp_path / "split"
        _prepare_split(img, lbl, split_dir)
        val_files = list((split_dir / "images" / "val").iterdir())
        assert len(val_files) >= 1

    def test_train_val_no_overlap(self, tmp_path):
        from scripts.train_cv_detector import _prepare_split
        img, lbl = self._make_dataset(tmp_path, n_seqs=10)
        split_dir = tmp_path / "split"
        _prepare_split(img, lbl, split_dir)
        train_stems = {p.stem for p in (split_dir / "images" / "train").iterdir()}
        val_stems   = {p.stem for p in (split_dir / "images" / "val").iterdir()}
        assert train_stems.isdisjoint(val_stems)

    def test_reuse_skips_recreation(self, tmp_path):
        from scripts.train_cv_detector import _prepare_split
        img, lbl = self._make_dataset(tmp_path, n_seqs=5)
        split_dir = tmp_path / "split"
        yaml1 = _prepare_split(img, lbl, split_dir)
        mtime1 = yaml1.stat().st_mtime
        yaml2 = _prepare_split(img, lbl, split_dir, force=False)
        assert yaml2.stat().st_mtime == mtime1     # not recreated

    def test_force_recreates(self, tmp_path):
        from scripts.train_cv_detector import _prepare_split
        import time
        img, lbl = self._make_dataset(tmp_path, n_seqs=5)
        split_dir = tmp_path / "split"
        yaml1 = _prepare_split(img, lbl, split_dir)
        time.sleep(0.05)
        yaml2 = _prepare_split(img, lbl, split_dir, force=True)
        assert yaml2.stat().st_mtime >= yaml1.stat().st_mtime


class TestMetricsCsv:
    def _make_csv(self, tmp_path: Path, map50=0.88, map50_95=0.60) -> Path:
        save_dir = tmp_path / "run"
        save_dir.mkdir()
        csv_path = save_dir / "results.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                         "metrics/precision(B)", "metrics/recall(B)"])
            w.writerow([1, 0.5, 0.3, 0.6, 0.55])
            w.writerow([2, map50, map50_95, 0.89, 0.84])
        return save_dir

    def test_reads_last_row(self, tmp_path):
        from scripts.train_cv_detector import _metrics_from_csv
        save_dir = self._make_csv(tmp_path, map50=0.88, map50_95=0.60)
        m = _metrics_from_csv(save_dir)
        assert abs(m["map50"] - 0.88) < 1e-5

    def test_returns_empty_if_no_csv(self, tmp_path):
        from scripts.train_cv_detector import _metrics_from_csv
        m = _metrics_from_csv(tmp_path)
        assert m == {}


class TestCliArgs:
    def test_defaults(self):
        from scripts.train_cv_detector import _parse_args
        args = _parse_args([])
        assert args.epochs == 50
        assert args.batch  == 16
        assert args.imgsz  == 640
        assert args.lr     == 0.01
        assert args.backbone == "yolov8n.pt"
        assert not args.dry_run
        assert not args.force
        assert not args.no_aug

    def test_overrides(self):
        from scripts.train_cv_detector import _parse_args
        args = _parse_args(["--epochs", "200", "--batch", "8", "--dry-run", "--force",
                             "--backbone", "yolov8s.pt", "--device", "cpu"])
        assert args.epochs == 200
        assert args.batch  == 8
        assert args.dry_run
        assert args.force
        assert args.backbone == "yolov8s.pt"
        assert args.device   == "cpu"
