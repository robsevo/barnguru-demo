"""Tests for Phase 16.1 — CV Training Data Pipeline.

Tests cover:
  - MOT → YOLO conversion (class map, normalization, ignore regions)
  - Augmentation transforms (flip correctness, warp/blur produce valid output)
  - Frame extractor helpers (sequence discovery, annotated frame filtering)
  - Dataset downloader helpers (dataset_present, list_sequences, list_videos)
  - Pipeline entry point (CLI argument parsing, dataset.yaml generation)

All tests are pure-Python / OpenCV; no actual NHL video data required.
"""

from __future__ import annotations

import configparser
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest

from data.cv_training.augmentor import (
    YoloBox,
    _brightness,
    _hflip_lbl,
    _load_labels,
    _perspective_matrix,
    _save_labels,
    _transform_boxes_perspective,
)
from data.cv_training.mot_to_yolo import (
    DEFAULT_CLASS_MAP,
    _parse_gt,
    _write_yolo,
)
from data.cv_training.frame_extractor import (
    _load_annotated_frame_ids,
    _parse_seqinfo,
)
from data.cv_training.dataset_downloader import (
    dataset_present,
    list_sequences,
    list_videos,
)
from scripts.build_cv_dataset import YOLO_CLASSES, _write_dataset_yaml


# ===========================================================================
# MOT → YOLO conversion
# ===========================================================================

class TestParseGt:
    def _make_gt(self, tmp_path: Path, lines: list[str]) -> Path:
        gt_dir = tmp_path / "seq" / "gt"
        gt_dir.mkdir(parents=True)
        gt = gt_dir / "gt.txt"
        gt.write_text("\n".join(lines) + "\n")
        return gt

    def test_basic_player_normalized(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "1,1,100,200,50,80,1,1,1.0",
        ])
        boxes = _parse_gt(gt, DEFAULT_CLASS_MAP, im_w=1000, im_h=1000)
        assert 1 in boxes
        cls, cx, cy, w, h = boxes[1][0]
        assert cls == 0            # player → player_home
        assert abs(cx - 0.125) < 1e-4   # (100 + 25) / 1000
        assert abs(cy - 0.24)  < 1e-4   # (200 + 40) / 1000
        assert abs(w  - 0.05)  < 1e-4
        assert abs(h  - 0.08)  < 1e-4

    def test_ignore_region_skipped(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "1,1,0,0,50,50,0,1,1.0",   # confidence=0 → ignore
        ])
        boxes = _parse_gt(gt, DEFAULT_CLASS_MAP, im_w=1000, im_h=1000)
        assert boxes == {}

    def test_puck_class_mapping(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "5,1,400,300,10,10,1,4,1.0",   # MOT class 4 → YOLO class 5 (puck)
        ])
        boxes = _parse_gt(gt, DEFAULT_CLASS_MAP, im_w=1000, im_h=1000)
        assert 5 in boxes
        assert boxes[5][0][0] == 5

    def test_referee_class_mapping(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "3,1,200,100,40,90,1,3,1.0",
        ])
        boxes = _parse_gt(gt, DEFAULT_CLASS_MAP, im_w=1920, im_h=1080)
        assert 3 in boxes
        assert boxes[3][0][0] == 4    # referee

    def test_tiny_box_filtered(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "1,1,100,100,2,2,1,1,1.0",   # w=2, h=2 < _MIN_BOX_PX=4
        ])
        boxes = _parse_gt(gt, DEFAULT_CLASS_MAP, im_w=1000, im_h=1000)
        assert boxes == {}

    def test_comment_lines_skipped(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "# This is a comment",
            "2,1,50,50,30,40,1,1,1.0",
        ])
        boxes = _parse_gt(gt, DEFAULT_CLASS_MAP, im_w=500, im_h=500)
        assert 2 in boxes

    def test_custom_class_map(self, tmp_path):
        gt = self._make_gt(tmp_path, [
            "1,1,100,100,50,60,1,99,1.0",   # unknown class 99
        ])
        custom_map = {**DEFAULT_CLASS_MAP, 99: 3}   # → goalie_away
        boxes = _parse_gt(gt, custom_map, im_w=1000, im_h=1000)
        assert 1 in boxes
        assert boxes[1][0][0] == 3

    def test_write_and_round_trip(self, tmp_path):
        lbl = tmp_path / "frame.txt"
        _write_yolo(lbl, [(0, 0.5, 0.5, 0.1, 0.2), (5, 0.8, 0.3, 0.05, 0.04)])
        content = lbl.read_text().strip().split("\n")
        assert len(content) == 2
        parts = content[0].split()
        assert parts[0] == "0"
        assert abs(float(parts[1]) - 0.5) < 1e-5


# ===========================================================================
# Augmentation
# ===========================================================================

class TestAugmentation:
    def _make_img(self, h=100, w=100):
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_hflip_label_cx_mirrored(self):
        boxes = [YoloBox(0, 0.3, 0.5, 0.1, 0.2)]
        flipped = _hflip_lbl(boxes, w=100, h=100)
        assert len(flipped) == 1
        assert abs(flipped[0].cx - 0.7) < 1e-6
        assert flipped[0].cy == 0.5
        assert flipped[0].cls == 0

    def test_hflip_label_symmetry(self):
        boxes = [YoloBox(1, 0.5, 0.5, 0.2, 0.3)]
        flipped = _hflip_lbl(boxes, w=200, h=100)
        assert abs(flipped[0].cx - 0.5) < 1e-6  # centre stays at 0.5

    def test_brightness_clamps(self):
        img = np.full((10, 10, 3), 240, dtype=np.uint8)
        bright = _brightness(img, +30)
        assert bright.max() <= 255
        dark = _brightness(img, -250)
        assert dark.min() >= 0

    def test_perspective_matrix_shape(self):
        M = _perspective_matrix(1920, 1080)
        assert M.shape == (3, 3)

    def test_perspective_matrix_deterministic(self):
        M1 = _perspective_matrix(1920, 1080, seed=7)
        M2 = _perspective_matrix(1920, 1080, seed=7)
        np.testing.assert_array_equal(M1, M2)

    def test_transform_boxes_perspective_keeps_valid_boxes(self):
        M = _perspective_matrix(100, 100, strength=0.01)
        boxes = [YoloBox(0, 0.5, 0.5, 0.1, 0.1)]
        result = _transform_boxes_perspective(boxes, M, w=100, h=100)
        assert len(result) == 1
        assert 0 <= result[0].cx <= 1
        assert 0 <= result[0].cy <= 1

    def test_save_load_labels_round_trip(self, tmp_path):
        path = tmp_path / "test.txt"
        boxes = [YoloBox(0, 0.5, 0.4, 0.1, 0.2), YoloBox(5, 0.9, 0.1, 0.02, 0.02)]
        _save_labels(path, boxes)
        loaded = _load_labels(path)
        assert len(loaded) == 2
        assert loaded[0].cls == 0
        assert abs(loaded[0].cx - 0.5) < 1e-5
        assert loaded[1].cls == 5


# ===========================================================================
# Frame extractor helpers
# ===========================================================================

class TestFrameExtractor:
    def test_load_annotated_frame_ids(self, tmp_path):
        gt = tmp_path / "gt.txt"
        gt.write_text("1,1,10,10,20,30,1,1,1\n3,2,50,50,20,20,1,1,1\n5,1,0,0,10,10,1,1,1\n")
        ids = _load_annotated_frame_ids(gt)
        assert ids == {1, 3, 5}

    def test_load_annotated_frame_ids_ignores_comments(self, tmp_path):
        gt = tmp_path / "gt.txt"
        gt.write_text("# comment\n2,1,0,0,10,10,1,1,1\n")
        ids = _load_annotated_frame_ids(gt)
        assert ids == {2}

    def test_parse_seqinfo_defaults(self, tmp_path):
        seq_dir = tmp_path / "seq1"
        seq_dir.mkdir()
        info = _parse_seqinfo(seq_dir / "seqinfo.ini", seq_dir)
        assert info["fps"] == 25.0
        assert info["im_width"] == 1920
        assert info["name"] == "seq1"

    def test_parse_seqinfo_reads_values(self, tmp_path):
        seq_dir = tmp_path / "seq2"
        seq_dir.mkdir()
        ini = seq_dir / "seqinfo.ini"
        ini.write_text(
            "[Sequence]\nname=TestSeq\nframeRate=30\nimWidth=1280\nimHeight=720\n"
        )
        info = _parse_seqinfo(ini, seq_dir)
        assert info["name"] == "TestSeq"
        assert info["fps"] == 30.0
        assert info["im_width"] == 1280
        assert info["im_height"] == 720


# ===========================================================================
# Dataset downloader helpers
# ===========================================================================

class TestDatasetDownloader:
    def test_dataset_present_empty_dir(self, tmp_path):
        assert not dataset_present(tmp_path)

    def test_dataset_present_with_gt(self, tmp_path):
        gt_dir = tmp_path / "seq1" / "gt"
        gt_dir.mkdir(parents=True)
        gt = gt_dir / "gt.txt"
        gt.write_text("1,1,0,0,10,10,1,1,1\n")
        assert dataset_present(tmp_path)

    def test_dataset_present_with_video(self, tmp_path):
        # Create a minimal valid mp4 header (just needs to exist + be non-empty)
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
        # No gt.txt, but has mp4
        assert dataset_present(tmp_path)

    def test_list_sequences_empty(self, tmp_path):
        seqs = list_sequences(tmp_path)
        assert seqs == []

    def test_list_sequences_finds_gt(self, tmp_path):
        for name in ("seq1", "seq2"):
            gt_dir = tmp_path / name / "gt"
            gt_dir.mkdir(parents=True)
            (gt_dir / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n")
        seqs = list_sequences(tmp_path)
        assert len(seqs) == 2

    def test_list_videos_empty(self, tmp_path):
        assert list_videos(tmp_path) == []

    def test_list_videos_finds_mp4(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"fake")
        (tmp_path / "b.avi").write_bytes(b"fake")
        videos = list_videos(tmp_path)
        assert len(videos) == 2


# ===========================================================================
# Pipeline entry point
# ===========================================================================

class TestPipelineYaml:
    def test_write_dataset_yaml(self, tmp_path):
        yaml_path = tmp_path / "dataset.yaml"
        _write_dataset_yaml(
            yaml_path,
            images_dir=tmp_path / "images",
            labels_dir=tmp_path / "labels",
            classes=YOLO_CLASSES,
        )
        content = yaml_path.read_text()
        assert "nc: 6" in content
        assert "player_home" in content
        assert "puck" in content
        assert "referee" in content

    def test_yolo_classes_count(self):
        assert len(YOLO_CLASSES) == 6

    def test_yolo_classes_order(self):
        assert YOLO_CLASSES[0] == "player_home"
        assert YOLO_CLASSES[1] == "player_away"
        assert YOLO_CLASSES[4] == "referee"
        assert YOLO_CLASSES[5] == "puck"

    def test_cli_parse_defaults(self):
        from scripts.build_cv_dataset import _parse_args
        args = _parse_args([])
        assert args.fps == 2.0
        assert not args.force
        assert not args.no_augment
        assert not args.skip_download

    def test_cli_parse_flags(self):
        from scripts.build_cv_dataset import _parse_args
        args = _parse_args(["--skip-download", "--fps", "5", "--no-augment", "--force"])
        assert args.skip_download
        assert args.fps == 5.0
        assert args.no_augment
        assert args.force
