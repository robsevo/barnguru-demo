//! Unit tests for Feature 16.6 — Kalman tracker.
//!
//! Tests cover:
//!   • IoU computation
//!   • Hungarian assignment algorithm
//!   • Kalman filter predict / update
//!   • Track lifecycle: Tentative → Confirmed → Lost → Deleted
//!   • Re-ID: lost track re-linked when reappearing within 2 s with IoU ≥ threshold
//!   • Full TrackerCore update loop

use gretzky_engine::tracker::{
    iou, hungarian, Kf, Det, TrackerCore, TrackState,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn det(x: f64, y: f64, w: f64, h: f64) -> Det {
    Det::from_tlwh(x, y, w, h, 1.0, 0)
}

fn tracker() -> TrackerCore {
    // n_init=3, max_lost=5, iou_threshold=0.3, reid_iou=0.3, fps=10
    TrackerCore::new(10.0, 3, 5, 0.3, 0.3)
}

// ── IoU ──────────────────────────────────────────────────────────────────────

#[test]
fn iou_identical_boxes_is_one() {
    assert!((iou((100.0, 100.0, 50.0, 50.0), (100.0, 100.0, 50.0, 50.0)) - 1.0).abs() < 1e-9);
}

#[test]
fn iou_non_overlapping_is_zero() {
    // box A: x 0..50, box B: x 100..150 — no overlap
    assert_eq!(iou((25.0, 25.0, 50.0, 50.0), (125.0, 25.0, 50.0, 50.0)), 0.0);
}

#[test]
fn iou_half_overlap() {
    // A: cx=50, cy=50, w=100, h=100 → [0,100]×[0,100]
    // B: cx=100, cy=50, w=100, h=100 → [50,150]×[0,100]
    // intersection: [50,100]×[0,100] = 50×100 = 5000
    // union: 10000 + 10000 - 5000 = 15000
    let got = iou((50.0, 50.0, 100.0, 100.0), (100.0, 50.0, 100.0, 100.0));
    let expected = 5000.0 / 15000.0;
    assert!((got - expected).abs() < 1e-9, "got {got}, expected {expected}");
}

#[test]
fn iou_contained_box() {
    // A: 100×100, B: 50×50 fully inside A
    // intersection = 50×50 = 2500, union = 10000+2500-2500 = 10000
    let got = iou((50.0, 50.0, 100.0, 100.0), (50.0, 50.0, 50.0, 50.0));
    let expected = 2500.0 / 10000.0;
    assert!((got - expected).abs() < 1e-9);
}

#[test]
fn iou_zero_area_box_returns_zero() {
    assert_eq!(iou((50.0, 50.0, 0.0, 0.0), (50.0, 50.0, 10.0, 10.0)), 0.0);
}

// ── Hungarian ────────────────────────────────────────────────────────────────

#[test]
fn hungarian_empty_cost_matrix() {
    let result = hungarian(&[]);
    assert!(result.is_empty());
}

#[test]
fn hungarian_single_cell() {
    let cost = vec![vec![3.0]];
    let asgn = hungarian(&cost);
    assert_eq!(asgn, vec![Some(0)]);
}

#[test]
fn hungarian_2x2_optimal() {
    // Optimal: row0→col1 (cost 2), row1→col0 (cost 1), total 3
    //          alt:    row0→col0 (cost 3), row1→col1 (cost 4), total 7
    let cost = vec![
        vec![3.0, 2.0],
        vec![1.0, 4.0],
    ];
    let asgn = hungarian(&cost);
    assert_eq!(asgn[0], Some(1));
    assert_eq!(asgn[1], Some(0));
}

#[test]
fn hungarian_3x3_optimal() {
    // Classic example from Wikipedia
    let cost = vec![
        vec![4.0, 1.0, 3.0],
        vec![2.0, 0.0, 5.0],
        vec![3.0, 2.0, 2.0],
    ];
    let asgn = hungarian(&cost);
    // Optimal: row0→col1(1), row1→col0(2), row2→col2(2) = 5
    let total: f64 = asgn.iter().enumerate()
        .map(|(i, c)| c.map(|j| cost[i][j]).unwrap_or(0.0))
        .sum();
    assert!((total - 5.0).abs() < 1e-9, "total cost = {total}");
}

#[test]
fn hungarian_more_rows_than_cols_some_unmatched() {
    // 3 tracks, 2 detections — one track must be None
    let cost = vec![
        vec![1.0, 5.0],
        vec![5.0, 1.0],
        vec![3.0, 3.0], // track 2 will be unmatched (padding)
    ];
    let asgn = hungarian(&cost);
    assert_eq!(asgn.len(), 3);
    // tracks 0 and 1 should get their best matches
    assert_eq!(asgn[0], Some(0));
    assert_eq!(asgn[1], Some(1));
    assert_eq!(asgn[2], None);
}

#[test]
fn hungarian_more_cols_than_rows() {
    // 2 tracks, 3 detections — one detection unmatched (but assignment still covers all tracks)
    let cost = vec![
        vec![1.0, 5.0, 3.0],
        vec![5.0, 1.0, 3.0],
    ];
    let asgn = hungarian(&cost);
    assert_eq!(asgn.len(), 2);
    assert_eq!(asgn[0], Some(0));
    assert_eq!(asgn[1], Some(1));
}

// ── Kalman filter ─────────────────────────────────────────────────────────────

#[test]
fn kf_initial_state_from_detection() {
    let kf = Kf::new(100.0, 200.0, 50.0, 80.0);
    assert!((kf.x[0] - 100.0).abs() < 1e-9, "cx");
    assert!((kf.x[1] - 200.0).abs() < 1e-9, "cy");
    assert!((kf.x[2]).abs()         < 1e-9, "vx = 0");
    assert!((kf.x[3]).abs()         < 1e-9, "vy = 0");
    assert!((kf.x[4] - 50.0).abs()  < 1e-9, "w");
    assert!((kf.x[5] - 80.0).abs()  < 1e-9, "h");
}

#[test]
fn kf_predict_advances_position_by_velocity() {
    let mut kf = Kf::new(100.0, 200.0, 50.0, 80.0);
    // Manually set velocity
    kf.x[2] = 5.0;  // vx
    kf.x[3] = -3.0; // vy
    kf.predict();
    // cx += vx, cy += vy (constant-velocity model)
    assert!((kf.x[0] - 105.0).abs() < 1e-6, "cx after predict = {}", kf.x[0]);
    assert!((kf.x[1] - 197.0).abs() < 1e-6, "cy after predict = {}", kf.x[1]);
    assert!((kf.x[2] - 5.0).abs()   < 1e-6, "vx unchanged");
    assert!((kf.x[3] - (-3.0)).abs()< 1e-6, "vy unchanged");
}

#[test]
fn kf_update_pulls_state_toward_measurement() {
    let mut kf = Kf::new(100.0, 100.0, 50.0, 50.0);
    // Predict once (no movement, vx=vy=0)
    kf.predict();
    let cx_before = kf.x[0];
    // Update with measurement shifted +20 in x
    kf.update(&[120.0, 100.0, 50.0, 50.0]);
    // State should move toward measurement
    assert!(kf.x[0] > cx_before, "cx should increase toward 120");
    assert!(kf.x[0] < 120.0,     "cx should not overshoot");
}

#[test]
fn kf_repeated_updates_converge() {
    let mut kf = Kf::new(0.0, 0.0, 40.0, 40.0);
    let z = [200.0, 300.0, 40.0, 40.0];
    for _ in 0..50 {
        kf.predict();
        kf.update(&z);
    }
    // After many updates the filter should be close to the true measurement
    assert!((kf.x[0] - 200.0).abs() < 1.0, "cx converged to 200, got {}", kf.x[0]);
    assert!((kf.x[1] - 300.0).abs() < 1.0, "cy converged to 300, got {}", kf.x[1]);
}

#[test]
fn kf_predicted_bbox_matches_state() {
    let mut kf = Kf::new(50.0, 60.0, 30.0, 40.0);
    kf.x[2] = 2.0; // vx
    kf.predict();
    let (cx, cy, w, h) = kf.predicted_bbox();
    assert!((cx - kf.x[0]).abs() < 1e-12);
    assert!((cy - kf.x[1]).abs() < 1e-12);
    assert!((w  - kf.x[4]).abs() < 1e-12);
    assert!((h  - kf.x[5]).abs() < 1e-12);
}

// ── TrackerCore — full lifecycle ──────────────────────────────────────────────

#[test]
fn tracker_empty_frame_no_tracks() {
    let mut tr = tracker();
    let results = tr.update(vec![]);
    assert!(results.is_empty());
    assert_eq!(tr.tracks.len(), 0);
}

#[test]
fn tracker_single_det_creates_tentative_track() {
    let mut tr = tracker();
    tr.update(vec![det(100.0, 100.0, 50.0, 50.0)]);
    assert_eq!(tr.tracks.len(), 1);
    assert_eq!(tr.tracks[0].state, TrackState::Tentative);
    assert_eq!(tr.confirmed_count(), 0);
}

#[test]
fn tracker_track_confirmed_after_n_init_hits() {
    let mut tr = tracker(); // n_init = 3
    for i in 0..3 {
        let x = 100.0 + i as f64; // tiny drift so IoU stays high
        tr.update(vec![det(x, 100.0, 50.0, 50.0)]);
    }
    assert_eq!(tr.confirmed_count(), 1, "track should be confirmed after 3 hits");
}

#[test]
fn tracker_track_goes_lost_on_miss() {
    let mut tr = tracker();
    for i in 0..3 {
        tr.update(vec![det(100.0 + i as f64, 100.0, 50.0, 50.0)]);
    }
    assert_eq!(tr.confirmed_count(), 1);
    // Miss one frame
    tr.update(vec![]);
    assert_eq!(tr.tracks.len(), 1);
    assert_eq!(tr.tracks[0].state, TrackState::Lost);
}

#[test]
fn tracker_lost_track_deleted_after_max_lost_frames() {
    let mut tr = tracker(); // max_lost = 5
    for i in 0..3 {
        tr.update(vec![det(100.0 + i as f64, 100.0, 50.0, 50.0)]);
    }
    // 6 consecutive misses — exceeds max_lost=5
    for _ in 0..6 {
        tr.update(vec![]);
    }
    assert_eq!(tr.tracks.len(), 0, "deleted track should be removed");
}

#[test]
fn tracker_tentative_deleted_on_first_miss() {
    let mut tr = tracker();
    tr.update(vec![det(100.0, 100.0, 50.0, 50.0)]); // tentative
    tr.update(vec![]);                                // miss → deleted
    assert_eq!(tr.tracks.len(), 0, "tentative track must vanish on first miss");
}

#[test]
fn tracker_multiple_dets_create_multiple_tracks() {
    let mut tr = tracker();
    for _ in 0..3 {
        tr.update(vec![
            det(100.0, 100.0, 40.0, 40.0),
            det(500.0, 300.0, 40.0, 40.0),
        ]);
    }
    assert_eq!(tr.confirmed_count(), 2);
}

#[test]
fn tracker_ids_are_stable_across_frames() {
    let mut tr = tracker();
    for _ in 0..3 {
        tr.update(vec![det(100.0, 100.0, 50.0, 50.0)]);
    }
    let id = tr.tracks[0].id;
    for _ in 0..5 {
        tr.update(vec![det(102.0, 100.0, 50.0, 50.0)]);
    }
    assert_eq!(tr.tracks[0].id, id, "track ID must not change");
}

#[test]
fn tracker_nonoverlapping_dets_get_different_ids() {
    let mut tr = tracker();
    // Two detections far apart — should become two separate confirmed tracks
    for _ in 0..3 {
        tr.update(vec![
            det(50.0,  50.0,  40.0, 40.0),
            det(600.0, 400.0, 40.0, 40.0),
        ]);
    }
    let ids: Vec<u32> = tr.tracks.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
}

// ── Re-ID ────────────────────────────────────────────────────────────────────

#[test]
fn tracker_reid_relinks_lost_track_within_window() {
    // fps=10, max_reid_window = ceil(10*2) = 20 frames
    let mut tr = TrackerCore::new(10.0, 3, 30, 0.3, 0.3);

    // Build a confirmed track
    for i in 0..3 {
        tr.update(vec![det(100.0 + i as f64, 100.0, 50.0, 50.0)]);
    }
    let original_id = tr.tracks[0].id;

    // Miss one frame → Lost
    tr.update(vec![]);
    assert_eq!(tr.tracks[0].state, TrackState::Lost);

    // Reappear immediately at nearly the same position (high IoU)
    let results = tr.update(vec![det(102.0, 100.0, 50.0, 50.0)]);

    // Should be confirmed again with the SAME id
    assert_eq!(tr.confirmed_count(), 1);
    assert_eq!(tr.tracks[0].id, original_id, "re-ID must preserve original track ID");
    assert_eq!(tr.tracks[0].state, TrackState::Confirmed);
    assert!(!results.is_empty());
}

#[test]
fn tracker_reid_does_not_relink_when_iou_too_low() {
    let mut tr = TrackerCore::new(10.0, 3, 30, 0.3, 0.8); // high reid_iou=0.8

    for i in 0..3 {
        tr.update(vec![det(100.0 + i as f64, 100.0, 50.0, 50.0)]);
    }
    let original_id = tr.tracks[0].id;

    tr.update(vec![]); // Lost

    // Detection at a shifted position with IoU < 0.8 — should NOT re-link
    let results = tr.update(vec![det(300.0, 300.0, 50.0, 50.0)]);

    // Original track still Lost, new track created as Tentative
    let still_lost = tr.tracks.iter().any(|t| t.id == original_id && t.state == TrackState::Lost);
    assert!(still_lost, "original track should remain lost (IoU too low for re-ID)");
    // No confirmed output yet (new track is only Tentative)
    assert!(results.is_empty());
}

#[test]
fn tracker_reset_clears_all_state() {
    let mut tr = tracker();
    for i in 0..3 {
        tr.update(vec![det(100.0 + i as f64, 100.0, 50.0, 50.0)]);
    }
    assert!(tr.confirmed_count() > 0);
    tr.reset();
    assert_eq!(tr.tracks.len(), 0);
    assert_eq!(tr.frame_count, 0);
    assert_eq!(tr.next_id, 1);
}

// ── TrackResult fields ────────────────────────────────────────────────────────

#[test]
fn tracker_result_fields_are_plausible() {
    let mut tr = tracker();
    for i in 0..3 {
        tr.update(vec![det(100.0 + i as f64, 200.0, 60.0, 80.0)]);
    }
    let results = tr.update(vec![det(103.0, 200.0, 60.0, 80.0)]);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.cx > 50.0  && r.cx < 200.0, "cx = {}", r.cx);
    assert!(r.cy > 100.0 && r.cy < 300.0, "cy = {}", r.cy);
    assert!(r.w  > 10.0  && r.w  < 200.0, "w = {}",  r.w);
    assert!(r.h  > 10.0  && r.h  < 300.0, "h = {}",  r.h);
    assert_eq!(r.class_id, 0);
}
