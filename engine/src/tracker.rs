//! Feature 16.6 — In-house Kalman multi-object tracker (Rust).
//!
//! State vector:  [cx, cy, vx, vy, w, h]   (centre x/y, pixel velocity, width, height)
//! Observation:   [cx, cy, w,  h]
//!
//! Track lifecycle
//! ───────────────
//!   Tentative  →  Confirmed  after `n_init` consecutive hits
//!   Confirmed  →  Lost       on first missed frame
//!   Lost       →  Deleted    after `max_lost` more missed frames
//!   Tentative  →  Deleted    on any missed frame
//!
//! Re-ID
//! ─────
//! A Lost track may be re-linked to a new detection when both:
//!   • it reappears within `fps * 2` frames of its last update, AND
//!   • IoU(last_bbox, new_det) ≥ reid_iou
//!
//! PyO3 export
//! ───────────
//!   from gretzky_engine import Tracker

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ── fixed-size matrix type aliases ──────────────────────────────────────────

type V6 = [f64; 6];
type V4 = [f64; 4];
type M6 = [[f64; 6]; 6]; //  6 rows × 6 cols
type M4 = [[f64; 4]; 4]; //  4 rows × 4 cols
type M4x6 = [[f64; 6]; 4]; //  4 rows × 6 cols  (H)
type M6x4 = [[f64; 4]; 6]; //  6 rows × 4 cols  (Hᵀ, K)

// ── matrix helpers ───────────────────────────────────────────────────────────

#[inline]
fn m6_mul(a: &M6, b: &M6) -> M6 {
    let mut r = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for k in 0..6 {
            if a[i][k] == 0.0 { continue; }
            for j in 0..6 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

#[inline]
fn m6_add(a: &M6, b: &M6) -> M6 {
    let mut r = [[0.0f64; 6]; 6];
    for i in 0..6 { for j in 0..6 { r[i][j] = a[i][j] + b[i][j]; } }
    r
}

#[inline]
fn m6_sub(a: &M6, b: &M6) -> M6 {
    let mut r = [[0.0f64; 6]; 6];
    for i in 0..6 { for j in 0..6 { r[i][j] = a[i][j] - b[i][j]; } }
    r
}

#[inline]
fn m6_transpose(a: &M6) -> M6 {
    let mut r = [[0.0f64; 6]; 6];
    for i in 0..6 { for j in 0..6 { r[i][j] = a[j][i]; } }
    r
}

#[inline]
fn m6_identity() -> M6 {
    let mut r = [[0.0f64; 6]; 6];
    for i in 0..6 { r[i][i] = 1.0; }
    r
}

/// 6×6 × 6-vector
#[inline]
fn m6_mul_v6(a: &M6, b: &V6) -> V6 {
    let mut r = [0.0f64; 6];
    for i in 0..6 { for k in 0..6 { r[i] += a[i][k] * b[k]; } }
    r
}

/// 4×6 × 6-vector → 4-vector
#[inline]
fn m4x6_mul_v6(a: &M4x6, b: &V6) -> V4 {
    let mut r = [0.0f64; 4];
    for i in 0..4 { for k in 0..6 { r[i] += a[i][k] * b[k]; } }
    r
}

/// transpose of 4×6 → 6×4
#[inline]
fn m4x6_transpose(a: &M4x6) -> M6x4 {
    let mut r = [[0.0f64; 4]; 6];
    for i in 0..4 { for j in 0..6 { r[j][i] = a[i][j]; } }
    r
}

/// 6×6 × 6×4 → 6×4  (P × Hᵀ)
#[inline]
fn m6_mul_m6x4(a: &M6, b: &M6x4) -> M6x4 {
    let mut r = [[0.0f64; 4]; 6];
    for i in 0..6 { for k in 0..6 { for j in 0..4 { r[i][j] += a[i][k] * b[k][j]; } } }
    r
}

/// 4×6 × 6×6 → 4×6  (H × P)
#[inline]
fn m4x6_mul_m6(a: &M4x6, b: &M6) -> M4x6 {
    let mut r = [[0.0f64; 6]; 4];
    for i in 0..4 { for k in 0..6 { for j in 0..6 { r[i][j] += a[i][k] * b[k][j]; } } }
    r
}

/// 4×6 × 6×4 → 4×4  (H × P × Hᵀ)
#[inline]
fn m4x6_mul_m6x4(a: &M4x6, b: &M6x4) -> M4 {
    let mut r = [[0.0f64; 4]; 4];
    for i in 0..4 { for k in 0..6 { for j in 0..4 { r[i][j] += a[i][k] * b[k][j]; } } }
    r
}

#[inline]
fn m4_add(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0f64; 4]; 4];
    for i in 0..4 { for j in 0..4 { r[i][j] = a[i][j] + b[i][j]; } }
    r
}

/// 6×4 × 4×4 → 6×4  (P×Hᵀ × S⁻¹ = K)
#[inline]
fn m6x4_mul_m4(a: &M6x4, b: &M4) -> M6x4 {
    let mut r = [[0.0f64; 4]; 6];
    for i in 0..6 { for k in 0..4 { for j in 0..4 { r[i][j] += a[i][k] * b[k][j]; } } }
    r
}

/// 6×4 × 4-vector → 6-vector  (K × innovation)
#[inline]
fn m6x4_mul_v4(a: &M6x4, b: &V4) -> V6 {
    let mut r = [0.0f64; 6];
    for i in 0..6 { for k in 0..4 { r[i] += a[i][k] * b[k]; } }
    r
}

/// 6×4 × 4×6 → 6×6  (K × H)
#[inline]
fn m6x4_mul_m4x6(a: &M6x4, b: &M4x6) -> M6 {
    let mut r = [[0.0f64; 6]; 6];
    for i in 0..6 { for k in 0..4 { for j in 0..6 { r[i][j] += a[i][k] * b[k][j]; } } }
    r
}

#[inline]
fn v6_add(a: &V6, b: &V6) -> V6 {
    let mut r = [0.0f64; 6];
    for i in 0..6 { r[i] = a[i] + b[i]; }
    r
}

#[inline]
fn v4_sub(a: &V4, b: &V4) -> V4 {
    let mut r = [0.0f64; 4];
    for i in 0..4 { r[i] = a[i] - b[i]; }
    r
}

/// 4×4 matrix inverse via Gauss-Jordan elimination with partial pivoting.
pub fn m4_inv(m: &M4) -> M4 {
    let mut a = [[0.0f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 { a[i][j] = m[i][j]; }
        a[i][i + 4] = 1.0;
    }
    for col in 0..4 {
        // partial pivot
        let mut max_row = col;
        for row in (col + 1)..4 {
            if a[row][col].abs() > a[max_row][col].abs() { max_row = row; }
        }
        a.swap(col, max_row);
        let pivot = a[col][col];
        assert!(pivot.abs() > 1e-15, "singular 4×4 matrix in Kalman update");
        for j in 0..8 { a[col][j] /= pivot; }
        for row in 0..4 {
            if row != col {
                let f = a[row][col];
                for j in 0..8 { a[row][j] -= f * a[col][j]; }
            }
        }
    }
    let mut inv = [[0.0f64; 4]; 4];
    for i in 0..4 { for j in 0..4 { inv[i][j] = a[i][j + 4]; } }
    inv
}

// ── Kalman filter constants ──────────────────────────────────────────────────

/// State transition (constant-velocity model, dt = 1 frame).
const KF_F: M6 = [
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

/// Measurement matrix: observation [cx, cy, w, h] from state [cx, cy, vx, vy, w, h].
const KF_H: M4x6 = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

/// Process noise (velocity terms larger → more dynamic objects).
const KF_Q: M6 = [
    [1.0,  0.0,  0.0,   0.0,   0.0,  0.0],
    [0.0,  1.0,  0.0,   0.0,   0.0,  0.0],
    [0.0,  0.0, 10.0,   0.0,   0.0,  0.0],
    [0.0,  0.0,  0.0,  10.0,   0.0,  0.0],
    [0.0,  0.0,  0.0,   0.0,   1.0,  0.0],
    [0.0,  0.0,  0.0,   0.0,   0.0,  1.0],
];

/// Measurement noise (size terms larger → detector bbox is noisy).
const KF_R: M4 = [
    [1.0, 0.0,  0.0,  0.0],
    [0.0, 1.0,  0.0,  0.0],
    [0.0, 0.0, 10.0,  0.0],
    [0.0, 0.0,  0.0, 10.0],
];

// ── Kalman filter ────────────────────────────────────────────────────────────

/// Constant-velocity Kalman filter. State: `[cx, cy, vx, vy, w, h]`.
pub struct Kf {
    pub x: V6, // state estimate
    pub p: M6, // state covariance
}

impl Kf {
    /// Initialise from a detection (zero velocity, large velocity uncertainty).
    pub fn new(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        let x = [cx, cy, 0.0, 0.0, w, h];
        let p = [
            [10.0,    0.0,    0.0,     0.0,     0.0,  0.0],
            [0.0,    10.0,    0.0,     0.0,     0.0,  0.0],
            [0.0,     0.0, 1_000.0,   0.0,     0.0,  0.0],
            [0.0,     0.0,    0.0,  1_000.0,   0.0,  0.0],
            [0.0,     0.0,    0.0,     0.0,    10.0,  0.0],
            [0.0,     0.0,    0.0,     0.0,     0.0, 10.0],
        ];
        Kf { x, p }
    }

    /// Predict one time step.
    pub fn predict(&mut self) {
        self.x = m6_mul_v6(&KF_F, &self.x);
        let fp   = m6_mul(&KF_F, &self.p);
        let ft   = m6_transpose(&KF_F);
        let fpft = m6_mul(&fp, &ft);
        self.p   = m6_add(&fpft, &KF_Q);
    }

    /// Incorporate a measurement `z = [cx, cy, w, h]`.
    pub fn update(&mut self, z: &V4) {
        let ht   = m4x6_transpose(&KF_H);          // 6×4
        let ph_t = m6_mul_m6x4(&self.p, &ht);      // 6×4
        let hp   = m4x6_mul_m6(&KF_H, &self.p);    // 4×6
        let hpht = m4x6_mul_m6x4(&hp, &ht);         // 4×4
        let s    = m4_add(&hpht, &KF_R);            // 4×4 innovation covariance
        let s_inv = m4_inv(&s);
        let k    = m6x4_mul_m4(&ph_t, &s_inv);      // 6×4 Kalman gain
        let z_hat = m4x6_mul_v6(&KF_H, &self.x);   // 4  predicted obs
        let inn   = v4_sub(z, &z_hat);              // 4  innovation
        self.x    = v6_add(&self.x, &m6x4_mul_v4(&k, &inn));
        let kh    = m6x4_mul_m4x6(&k, &KF_H);      // 6×6
        let i_kh  = m6_sub(&m6_identity(), &kh);
        self.p    = m6_mul(&i_kh, &self.p);
    }

    /// Predicted bounding box `(cx, cy, w, h)` from current state.
    pub fn predicted_bbox(&self) -> (f64, f64, f64, f64) {
        (self.x[0], self.x[1], self.x[4], self.x[5])
    }
}

// ── IoU ─────────────────────────────────────────────────────────────────────

/// Intersection-over-Union for two bounding boxes `(cx, cy, w, h)`.
pub fn iou(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    let ax1 = a.0 - a.2 * 0.5;
    let ay1 = a.1 - a.3 * 0.5;
    let ax2 = a.0 + a.2 * 0.5;
    let ay2 = a.1 + a.3 * 0.5;

    let bx1 = b.0 - b.2 * 0.5;
    let by1 = b.1 - b.3 * 0.5;
    let bx2 = b.0 + b.2 * 0.5;
    let by2 = b.1 + b.3 * 0.5;

    let inter_w = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let inter_h = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter   = inter_w * inter_h;
    let union   = a.2 * a.3 + b.2 * b.3 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

// ── Hungarian (Jonker-Volgenant successive shortest paths) ───────────────────

/// Minimum-cost assignment via the JV / Kuhn-Munkres algorithm.
///
/// `cost[i][j]` = cost of assigning row `i` to column `j`.
/// Returns `assignment[i]` = `Some(col)` or `None` (unmatched).
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = cost.len();
    if rows == 0 { return vec![]; }
    let cols = cost[0].len();
    if cols == 0 { return vec![None; rows]; }
    let n = rows.max(cols);

    // Pad to n×n (pad cells get cost 0 so they're never preferred over real matches)
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..rows {
        for j in 0..cols {
            a[i][j] = cost[i][j];
        }
    }

    let mut u   = vec![0.0f64; n + 1]; // row potentials   (1-indexed)
    let mut v   = vec![0.0f64; n + 1]; // column potentials (1-indexed)
    let mut p   = vec![0usize; n + 1]; // p[j] = row assigned to col j; 0 = none
    let mut way = vec![0usize; n + 1]; // augmenting-path predecessor

    for i in 1..=n {
        p[0] = i;
        let mut j0: usize = 0;
        let mut minv = vec![f64::MAX; n + 1];
        let mut used = vec![false;    n + 1];

        // Find shortest augmenting path from row i
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::MAX;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Update potentials
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 { break; } // found free column — augmenting path complete
        }

        // Augment along path
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }

    // Invert p: find which column each real row was assigned to
    let mut out = vec![None; rows];
    for j in 1..=n {
        let row = p[j];
        if row > 0 && row <= rows && j <= cols {
            out[row - 1] = Some(j - 1);
        }
    }
    out
}

// ── Detection ────────────────────────────────────────────────────────────────

/// A single detection from the YOLO model (pixel coordinates, top-left origin).
#[derive(Clone)]
pub struct Det {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub class_id: u8,
}

impl Det {
    /// Construct from top-left `(x, y, w, h)`.
    pub fn from_tlwh(x: f64, y: f64, w: f64, h: f64, confidence: f64, class_id: u8) -> Self {
        Det { cx: x + w * 0.5, cy: y + h * 0.5, w, h, confidence, class_id }
    }
}

// ── Track result returned to caller ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrackResult {
    pub id: u32,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub vx: f64,
    pub vy: f64,
    pub class_id: u8,
}

// ── Track lifecycle ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost,
    Deleted,
}

pub struct Track {
    pub id: u32,
    pub kf: Kf,
    pub state: TrackState,
    pub hits: u32,
    pub misses: u32,
    pub class_id: u8,
    pub last_frame: u32,
    /// Last observed bbox `(cx, cy, w, h)` — used for re-ID.
    pub last_bbox: (f64, f64, f64, f64),
}

impl Track {
    fn new(id: u32, det: &Det, frame: u32) -> Self {
        Track {
            id,
            kf: Kf::new(det.cx, det.cy, det.w, det.h),
            state: TrackState::Tentative,
            hits: 1,
            misses: 0,
            class_id: det.class_id,
            last_frame: frame,
            last_bbox: (det.cx, det.cy, det.w, det.h),
        }
    }

    fn predict(&mut self) {
        self.kf.predict();
    }

    fn update(&mut self, det: &Det, frame: u32, n_init: u32) {
        self.kf.update(&[det.cx, det.cy, det.w, det.h]);
        self.hits += 1;
        self.misses = 0;
        self.last_frame = frame;
        self.last_bbox = (det.cx, det.cy, det.w, det.h);

        match self.state {
            TrackState::Tentative => {
                if self.hits >= n_init {
                    self.state = TrackState::Confirmed;
                }
            }
            TrackState::Lost => {
                self.state = TrackState::Confirmed; // re-ID succeeded
            }
            _ => {}
        }
    }

    fn mark_missed(&mut self, max_lost: u32) {
        self.misses += 1;
        match self.state {
            TrackState::Tentative => {
                self.state = TrackState::Deleted;
            }
            TrackState::Confirmed => {
                self.state = TrackState::Lost;
            }
            TrackState::Lost => {
                if self.misses > max_lost {
                    self.state = TrackState::Deleted;
                }
            }
            TrackState::Deleted => {}
        }
    }
}

// ── TrackerCore — pure Rust, no PyO3 ────────────────────────────────────────

/// Multi-object Kalman tracker (pure Rust — testable without a Python runtime).
pub struct TrackerCore {
    pub tracks: Vec<Track>,
    pub next_id: u32,
    pub frame_count: u32,
    pub fps: f64,
    pub n_init: u32,
    pub max_lost: u32,
    pub iou_threshold: f64,
    pub reid_iou: f64,
}

impl TrackerCore {
    pub fn new(
        fps: f64,
        n_init: u32,
        max_lost: u32,
        iou_threshold: f64,
        reid_iou: f64,
    ) -> Self {
        TrackerCore {
            tracks: vec![],
            next_id: 1,
            frame_count: 0,
            fps,
            n_init,
            max_lost,
            iou_threshold,
            reid_iou,
        }
    }

    /// Process one frame of detections.  Returns results for all **Confirmed** tracks.
    pub fn update(&mut self, dets: Vec<Det>) -> Vec<TrackResult> {
        self.frame_count += 1;

        // ── 1. Predict all existing tracks ──────────────────────────────────
        for t in &mut self.tracks {
            t.predict();
        }

        // ── 2. Partition tracks by state ────────────────────────────────────
        let active_idx: Vec<usize> = self.tracks.iter().enumerate()
            .filter(|(_, t)| t.state == TrackState::Confirmed || t.state == TrackState::Tentative)
            .map(|(i, _)| i)
            .collect();

        let lost_idx: Vec<usize> = self.tracks.iter().enumerate()
            .filter(|(_, t)| t.state == TrackState::Lost)
            .map(|(i, _)| i)
            .collect();

        // ── 3. Match active tracks → detections (IoU cost, Hungarian) ───────
        let mut used = vec![false; dets.len()];

        if !active_idx.is_empty() && !dets.is_empty() {
            let cost: Vec<Vec<f64>> = active_idx.iter().map(|&ti| {
                let bbox = self.tracks[ti].kf.predicted_bbox();
                dets.iter()
                    .map(|d| 1.0 - iou(bbox, (d.cx, d.cy, d.w, d.h)))
                    .collect()
            }).collect();

            let assignment = hungarian(&cost);

            for (ai, col_opt) in assignment.iter().enumerate() {
                let ti = active_idx[ai];
                match col_opt {
                    Some(di) if (1.0 - cost[ai][*di]) >= self.iou_threshold => {
                        self.tracks[ti].update(&dets[*di], self.frame_count, self.n_init);
                        used[*di] = true;
                    }
                    _ => {
                        self.tracks[ti].mark_missed(self.max_lost);
                    }
                }
            }
        } else {
            for &ti in &active_idx {
                self.tracks[ti].mark_missed(self.max_lost);
            }
        }

        // ── 4. Re-ID: match unmatched dets → lost tracks ─────────────────
        let max_reid_frames = (self.fps * 2.0).ceil() as u32;
        let unmatched_dets: Vec<usize> = (0..dets.len()).filter(|&i| !used[i]).collect();
        let mut relinked_lost: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        if !lost_idx.is_empty() && !unmatched_dets.is_empty() {
            let reid_cost: Vec<Vec<f64>> = lost_idx.iter().map(|&ti| {
                let frames_gone = self.frame_count.saturating_sub(self.tracks[ti].last_frame);
                let bbox = self.tracks[ti].last_bbox;
                unmatched_dets.iter().map(|&di| {
                    if frames_gone <= max_reid_frames {
                        1.0 - iou(bbox, (dets[di].cx, dets[di].cy, dets[di].w, dets[di].h))
                    } else {
                        2.0 // > 1.0 → IoU penalty, guaranteed below any reid_iou
                    }
                }).collect()
            }).collect();

            let reid_asgn = hungarian(&reid_cost);

            for (li, col_opt) in reid_asgn.iter().enumerate() {
                if let Some(ui) = col_opt {
                    let reid_iou_val = 1.0 - reid_cost[li][*ui];
                    if reid_iou_val >= self.reid_iou {
                        let ti = lost_idx[li];
                        let di = unmatched_dets[*ui];
                        self.tracks[ti].update(&dets[di], self.frame_count, self.n_init);
                        used[di] = true;
                        relinked_lost.insert(ti);
                    }
                }
            }
        }

        // Advance miss counter for lost tracks that were NOT re-linked this frame
        for &ti in &lost_idx {
            if !relinked_lost.contains(&ti) {
                self.tracks[ti].mark_missed(self.max_lost);
            }
        }

        // ── 5. Spawn new tracks for leftover detections ──────────────────
        for (di, &already_used) in used.iter().enumerate() {
            if !already_used {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(Track::new(id, &dets[di], self.frame_count));
            }
        }

        // ── 6. Reap deleted tracks ───────────────────────────────────────
        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // ── 7. Emit confirmed tracks ─────────────────────────────────────
        self.tracks.iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .map(|t| TrackResult {
                id: t.id,
                cx: t.kf.x[0],
                cy: t.kf.x[1],
                vx: t.kf.x[2],
                vy: t.kf.x[3],
                w:  t.kf.x[4],
                h:  t.kf.x[5],
                class_id: t.class_id,
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
        self.frame_count = 0;
    }

    pub fn confirmed_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.state == TrackState::Confirmed).count()
    }

    pub fn active_count(&self) -> usize {
        self.tracks.iter().filter(|t| {
            t.state == TrackState::Confirmed || t.state == TrackState::Tentative
        }).count()
    }
}

// ── Tracker — PyO3 wrapper ───────────────────────────────────────────────────

/// Multi-object Kalman tracker exposed to Python.
///
/// ```python
/// from gretzky_engine import Tracker
///
/// tracker = Tracker(fps=30.0, n_init=3, max_lost=30, iou_threshold=0.3, reid_iou=0.3)
///
/// # detections: list of [x, y, w, h, confidence, class_id]  (top-left origin)
/// results = tracker.update(detections)
/// # results: list of dicts {id, cx, cy, w, h, vx, vy, state, class_id}
/// ```
#[pyclass]
pub struct Tracker {
    inner: TrackerCore,
}

#[pymethods]
impl Tracker {
    /// Create a new tracker.
    ///
    /// Args:
    ///     fps:            Frames per second (used for the 2-second re-ID window). Default 30.
    ///     n_init:         Consecutive hits required to confirm a track. Default 3.
    ///     max_lost:       Missed frames before a Lost track is deleted. Default 30.
    ///     iou_threshold:  Minimum IoU to accept a track-detection match. Default 0.3.
    ///     reid_iou:       Minimum IoU to re-link a Lost track. Default 0.3.
    #[new]
    #[pyo3(signature = (fps=30.0, n_init=3, max_lost=30, iou_threshold=0.3, reid_iou=0.3))]
    pub fn new(
        fps: f64,
        n_init: u32,
        max_lost: u32,
        iou_threshold: f64,
        reid_iou: f64,
    ) -> Self {
        Tracker { inner: TrackerCore::new(fps, n_init, max_lost, iou_threshold, reid_iou) }
    }

    /// Process one frame of detections.
    ///
    /// Args:
    ///     detections: list of ``[x, y, w, h, confidence, class_id]``
    ///                 where ``(x, y)`` is the **top-left** corner in pixel coords.
    ///
    /// Returns:
    ///     List of dicts for every **confirmed** track.
    ///     Keys: `id`, `cx`, `cy`, `w`, `h`, `vx`, `vy`, `state`, `class_id`.
    pub fn update(
        &mut self,
        py: Python<'_>,
        detections: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dets: Vec<Det> = detections.iter().map(|row| {
            Det::from_tlwh(
                row.first().copied().unwrap_or(0.0),
                row.get(1).copied().unwrap_or(0.0),
                row.get(2).copied().unwrap_or(0.0),
                row.get(3).copied().unwrap_or(0.0),
                row.get(4).copied().unwrap_or(1.0),
                row.get(5).copied().unwrap_or(0.0) as u8,
            )
        }).collect();

        let results = self.inner.update(dets);

        results.iter().map(|r| {
            let d = PyDict::new(py);
            d.set_item("id",       r.id)?;
            d.set_item("cx",       r.cx)?;
            d.set_item("cy",       r.cy)?;
            d.set_item("w",        r.w)?;
            d.set_item("h",        r.h)?;
            d.set_item("vx",       r.vx)?;
            d.set_item("vy",       r.vy)?;
            d.set_item("state",    "confirmed")?;
            d.set_item("class_id", r.class_id)?;
            Ok(d.into_any().unbind())
        }).collect()
    }

    /// Clear all tracks and reset frame counter.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Number of currently confirmed tracks.
    pub fn confirmed_count(&self) -> usize {
        self.inner.confirmed_count()
    }

    /// Number of active tracks (confirmed + tentative).
    pub fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Current frame count.
    pub fn frame_count(&self) -> u32 {
        self.inner.frame_count
    }
}
