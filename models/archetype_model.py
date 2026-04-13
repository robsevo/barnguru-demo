"""Player Archetype Clustering — Feature 2.11.

K-means (8–12 archetypes) on player RAPM + rate stats.
Cluster centroids are exposed as a Bob-auditable DataFrame so Bob can
inspect each archetype's centroid values and relabel if needed.

Used by Feature 2.3 (Matchup Efficiency Matrix) to fill gaps when
pair history < 10 GP: fall back to archetype-vs-archetype mean xGF%.

Outputs
-------
``player_archetypes_{season}.parquet`` — one row per player:
    cluster_id      integer cluster label (0 … k-1)
    archetype       human-readable cluster name
    distance        Euclidean distance to cluster centroid (fit quality)

``archetype_centroids_{season}.parquet`` — one row per cluster:
    cluster_id, archetype, off_rapm, def_rapm, …  (centroid values)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "archetype_v3"
DEFAULT_K     = 10      # number of archetypes
MIN_K         = 8
MAX_K         = 12
MIN_PLAYERS   = 20      # minimum players to fit meaningfully

# Feature names and their indices into the feature matrix
SKATER_FEATURE_NAMES: list[str] = [
    "off_rapm",       # EV offensive RAPM
    "def_rapm",       # EV defensive RAPM
    "pp_rapm",        # PP RAPM (or 0.0 proxy)
    "pk_rapm",        # PK RAPM (or 0.0 proxy)
    "xg_per_game",    # individual expected goals per game (from xg_finishing)
    "shots_per_game", # individual shots on goal per game
    "log1p_gp",       # log1p(games_played) — sample size signal
]

# 10 archetype names assigned by composite-score rank
# Bob can relabel via rename_archetype() if the auto-name misses
_DEFAULT_ARCHETYPE_NAMES: list[str] = [
    "Elite Two-Way",
    "Elite Scorer",
    "Strong Two-Way",
    "Defensive Anchor",
    "PP Specialist",
    "Middle-Six Forward",
    "Offensive D",
    "Defensive Specialist",
    "Depth Forward",
    "4th Line",
]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

PLAYER_ARCHETYPE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":    pl.Int64,
    "player_name":  pl.Utf8,
    "season":       pl.Int64,
    "cluster_id":   pl.Int64,
    "archetype":    pl.Utf8,
    "distance":     pl.Float64,
    "model_version": pl.Utf8,
}

CENTROID_SCHEMA: dict[str, pl.DataType] = {
    "cluster_id": pl.Int64,
    "archetype":  pl.Utf8,
    "n_players":  pl.Int64,
    **{f: pl.Float64 for f in SKATER_FEATURE_NAMES},
}


# ---------------------------------------------------------------------------
# Auto-labelling helpers
# ---------------------------------------------------------------------------

def _assign_archetype_labels(
    centroids_raw: np.ndarray,   # (k, n_features) in original feature space
    feature_names: list[str],
    k: int,
) -> list[str]:
    """Assign Bob-auditable archetype names to k clusters.

    Ranking strategy:
    1. Compute composite score = 0.6 × off_rapm + 0.4 × def_rapm.
    2. Sort clusters highest→lowest composite score.
    3. Assign names from _DEFAULT_ARCHETYPE_NAMES (scaled to k).
    4. Optionally override specific clusters with specialty names
       (PP Specialist, Defensive Anchor) based on feature outliers.
    """
    try:
        off_idx = feature_names.index("off_rapm")
        def_idx = feature_names.index("def_rapm")
    except ValueError:
        return [f"Archetype_{i}" for i in range(k)]

    # Include individual xg rate in composite if available — this stabilises
    # rankings for players with small RAPM samples (e.g. injury-shortened seasons).
    xg_idx = feature_names.index("xg_per_game") if "xg_per_game" in feature_names else None

    if xg_idx is not None:
        # Normalise xg_per_game to same scale as RAPM values before combining
        xg_vals = centroids_raw[:, xg_idx]
        xg_range = xg_vals.max() - xg_vals.min()
        xg_norm = (xg_vals - xg_vals.min()) / (xg_range if xg_range > 0 else 1.0)
        # Map normalised xg to a ±1 range centred at median to mix with RAPM
        xg_scaled = (xg_norm - 0.5) * 2.0
        composite = (
            0.45 * centroids_raw[:, off_idx]
            - 0.30 * centroids_raw[:, def_idx]   # negate: rapm_ev_def < 0 = better defense
            + 0.25 * xg_scaled
        )
    else:
        composite = (
            0.6 * centroids_raw[:, off_idx]
            - 0.4 * centroids_raw[:, def_idx]    # negate: rapm_ev_def < 0 = better defense
        )
    rank_order = sorted(range(k), key=lambda c: composite[c], reverse=True)

    # Scale 10 canonical names to k clusters
    names: list[str] = [""] * k
    for rank, cluster_id in enumerate(rank_order):
        label_idx = int(round(rank * (len(_DEFAULT_ARCHETYPE_NAMES) - 1) / max(k - 1, 1)))
        names[cluster_id] = _DEFAULT_ARCHETYPE_NAMES[label_idx]

    # Specialty overrides
    # PP Specialist: has highest pp_rapm but not highest off_rapm
    if "pp_rapm" in feature_names:
        pp_idx = feature_names.index("pp_rapm")
        pp_scores = centroids_raw[:, pp_idx]
        best_pp_cluster = int(np.argmax(pp_scores))
        # Only relabel if it's not already the top offensive cluster
        if best_pp_cluster != rank_order[0] and best_pp_cluster != rank_order[1]:
            names[best_pp_cluster] = "PP Specialist"

    # Defensive Anchor: most negative def_rapm (rapm_ev_def < 0 = better defense)
    # AND below-median off_rapm — pure defensive specialist cluster
    med_off = np.median(centroids_raw[:, off_idx])
    def_scores = centroids_raw[:, def_idx]
    best_def_cluster = int(np.argmin(def_scores))   # argmin: most negative = best defense
    if centroids_raw[best_def_cluster, off_idx] < med_off:
        names[best_def_cluster] = "Defensive Anchor"

    return names


# ---------------------------------------------------------------------------
# PlayerArchetypeModel
# ---------------------------------------------------------------------------

class PlayerArchetypeModel:
    """K-means player archetype clustering (Feature 2.11).

    Usage::

        model = PlayerArchetypeModel(k=10)
        metrics = model.fit(rapm_df, season=2025)
        assignments = model.predict(rapm_df, season=2025)
        centroids_df = model.centroids_dataframe()
        model.save(Path("~/.gretzky/data/archetypes/archetype_model.pkl"))
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        random_state: int = 42,
        n_init: int = 20,
    ) -> None:
        if not MIN_K <= k <= MAX_K:
            raise ValueError(f"k must be between {MIN_K} and {MAX_K}, got {k}")
        self._k = k
        self._random_state = random_state
        self._n_init = n_init
        self._kmeans: KMeans | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = list(SKATER_FEATURE_NAMES)
        self._archetype_names: list[str] = []
        self._centroids_original: np.ndarray | None = None   # (k, n_features) unscaled
        self._cluster_sizes: np.ndarray | None = None
        self._version = MODEL_VERSION
        self._custom_names: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(
        self, rapm_df: pl.DataFrame
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """Extract feature matrix from RAPM DataFrame.

        Returns:
            (X float32, player_ids list, player_names list)
        """
        required = {"player_id"}
        missing = required - set(rapm_df.columns)
        if missing:
            raise ValueError(f"rapm_df missing columns: {sorted(missing)}")

        df = rapm_df

        # Build feature matrix with safe fallbacks for missing columns
        def _col_or_zero(name: str) -> pl.Series:
            if name in df.columns:
                return df[name].cast(pl.Float64).fill_null(0.0)
            return pl.Series(name, [0.0] * len(df), dtype=pl.Float64)

        # Support both naming conventions: archetype names and RAPM parquet names
        def _col_or_zero_alias(*names: str) -> pl.Series:
            for n in names:
                if n in df.columns:
                    return df[n].cast(pl.Float64).fill_null(0.0)
            return pl.Series(names[0], [0.0] * len(df), dtype=pl.Float64)

        cols = {
            "off_rapm":       _col_or_zero_alias("off_rapm", "rapm_ev_off"),
            "def_rapm":       _col_or_zero_alias("def_rapm", "rapm_ev_def"),
            "pp_rapm":        _col_or_zero_alias("pp_rapm", "rapm_pp"),
            "pk_rapm":        _col_or_zero_alias("pk_rapm", "rapm_pk"),
            # Individual production features — enriched from xg_finishing by the
            # training script.  Fall back to on-ice context columns if absent.
            "xg_per_game":    _col_or_zero_alias("xg_per_game", "xg_sum_per_game", "xga_60"),
            "shots_per_game": _col_or_zero_alias("shots_per_game", "shots_per60"),
        }
        gp_col = _col_or_zero_alias("games_played", "gp")
        log1p_gp = pl.Series(
            "log1p_gp",
            np.log1p(gp_col.to_numpy()),
            dtype=pl.Float64,
        )

        X_raw = np.column_stack([
            cols["off_rapm"].to_numpy(),
            cols["def_rapm"].to_numpy(),
            cols["pp_rapm"].to_numpy(),
            cols["pk_rapm"].to_numpy(),
            cols["xg_per_game"].to_numpy(),
            cols["shots_per_game"].to_numpy(),
            log1p_gp.to_numpy(),
        ]).astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0)

        player_ids   = df["player_id"].cast(pl.Int64).to_list()
        player_names = (
            df["player_name"].to_list()
            if "player_name" in df.columns
            else [""] * len(df)
        )
        return X_raw.astype(np.float32), player_ids, player_names

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        rapm_df: pl.DataFrame,
        season: int,
    ) -> dict[str, Any]:
        """Fit k-means archetypes on player RAPM features.

        Args:
            rapm_df: DataFrame with columns including player_id and any
                     subset of: off_rapm, def_rapm, pp_rapm, pk_rapm,
                     xgf_per60, shots_per60, games_played.
            season:  NHL season start-year (for logging only).

        Returns:
            {"inertia": float, "n_players": int, "silhouette": float or None}
        """
        X, _, _ = self._build_features(rapm_df)

        if len(X) < MIN_PLAYERS:
            warnings.warn(
                f"Season {season}: only {len(X)} players — archetype model may be unstable.",
                DataMissingWarning,
                stacklevel=2,
            )

        if len(X) < self._k:
            raise ValueError(
                f"Cannot fit {self._k} clusters on only {len(X)} players."
            )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X.astype(np.float64))

        self._kmeans = KMeans(
            n_clusters    = self._k,
            random_state  = self._random_state,
            n_init        = self._n_init,
        )
        self._kmeans.fit(X_scaled)

        # Invert scaling on centroids for interpretability
        self._centroids_original = self._scaler.inverse_transform(
            self._kmeans.cluster_centers_
        )

        # Count cluster sizes
        labels = self._kmeans.labels_
        self._cluster_sizes = np.bincount(labels, minlength=self._k)

        # Auto-assign archetype names
        self._archetype_names = _assign_archetype_labels(
            self._centroids_original,
            self._feature_names,
            self._k,
        )

        inertia = float(self._kmeans.inertia_)

        # Silhouette (optional, import guard)
        silhouette: float | None = None
        try:
            from sklearn.metrics import silhouette_score
            silhouette = float(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X))))
        except Exception:
            pass

        return {"inertia": inertia, "n_players": len(X), "silhouette": silhouette}

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        rapm_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Assign archetype to every player in rapm_df.

        Returns:
            DataFrame matching PLAYER_ARCHETYPE_SCHEMA.

        Raises:
            RuntimeError: if model not fitted.
        """
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X, player_ids, player_names = self._build_features(rapm_df)
        X_scaled = self._scaler.transform(X.astype(np.float64))  # sklearn requires float64
        cluster_ids = self._kmeans.predict(X_scaled)

        # Distance to assigned centroid
        centers = self._kmeans.cluster_centers_
        distances = np.linalg.norm(X_scaled - centers[cluster_ids], axis=1)

        archetypes = [
            self._custom_names.get(int(c), self._archetype_names[int(c)])
            for c in cluster_ids
        ]

        return pl.DataFrame({
            "player_id":    pl.Series(player_ids,                      dtype=pl.Int64),
            "player_name":  pl.Series(player_names,                    dtype=pl.Utf8),
            "season":       pl.Series([season] * len(X),               dtype=pl.Int64),
            "cluster_id":   pl.Series(cluster_ids.tolist(),            dtype=pl.Int64),
            "archetype":    pl.Series(archetypes,                      dtype=pl.Utf8),
            "distance":     pl.Series(distances.tolist(),              dtype=pl.Float64),
            "model_version": pl.Series([self._version] * len(X),      dtype=pl.Utf8),
        })

    # ------------------------------------------------------------------
    # Centroids (Bob-auditable)
    # ------------------------------------------------------------------

    def centroids_dataframe(self) -> pl.DataFrame:
        """Return cluster centroids in original feature space.

        This is the primary Bob-auditable output: each row is one
        archetype with its mean feature values.  Use this to verify
        that e.g. "Elite Two-Way" has high off_rapm AND high def_rapm.
        """
        if self._centroids_original is None or self._kmeans is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        rows: dict[str, list] = {
            "cluster_id": list(range(self._k)),
            "archetype":  [
                self._custom_names.get(i, self._archetype_names[i])
                for i in range(self._k)
            ],
            "n_players":  self._cluster_sizes.tolist(),
        }
        for j, fname in enumerate(self._feature_names):
            rows[fname] = self._centroids_original[:, j].tolist()

        return pl.DataFrame(rows).sort("cluster_id")

    def rename_archetype(self, cluster_id: int, new_name: str) -> None:
        """Override auto-generated archetype name for a cluster.

        Bob can call this after inspecting centroids_dataframe() to
        relabel any cluster that doesn't match the auto-assigned name.
        """
        if self._kmeans is None:
            raise RuntimeError("Model not fitted.")
        if not (0 <= cluster_id < self._k):
            raise ValueError(f"cluster_id must be 0–{self._k-1}, got {cluster_id}")
        self._custom_names[cluster_id] = new_name

    def archetype_names(self) -> dict[int, str]:
        """Return {cluster_id: archetype_name} mapping."""
        if not self._archetype_names:
            raise RuntimeError("Model not fitted.")
        return {
            i: self._custom_names.get(i, self._archetype_names[i])
            for i in range(self._k)
        }

    # ------------------------------------------------------------------
    # Matchup gap-fill helper (feeds Feature 2.3)
    # ------------------------------------------------------------------

    def archetype_matchup_prior(
        self,
        assignments_df: pl.DataFrame,
        observed_pair_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute archetype-level mean xGF% for gap-filling.

        When a player pair has GP < 10 in observed_pair_df, the archetype
        matchup prior is used as the baseline.

        Args:
            assignments_df:  output of predict() with player_id + cluster_id.
            observed_pair_df: pair chemistry data with player_a_id, player_b_id,
                              model_xgf_pct columns.

        Returns:
            DataFrame with cluster_a, cluster_b, mean_xgf_pct — archetype
            level prior for each archetype pair.
        """
        if "player_a_id" not in observed_pair_df.columns:
            return pl.DataFrame(
                schema={"cluster_a": pl.Int64, "cluster_b": pl.Int64, "mean_xgf_pct": pl.Float64}
            )

        id_to_cluster = dict(
            zip(
                assignments_df["player_id"].to_list(),
                assignments_df["cluster_id"].to_list(),
            )
        )

        # Join cluster ids onto pairs
        pairs = observed_pair_df.with_columns([
            pl.col("player_a_id")
            .map_elements(lambda x: id_to_cluster.get(x, -1), return_dtype=pl.Int64)
            .alias("cluster_a"),
            pl.col("player_b_id")
            .map_elements(lambda x: id_to_cluster.get(x, -1), return_dtype=pl.Int64)
            .alias("cluster_b"),
        ]).filter(
            (pl.col("cluster_a") >= 0) & (pl.col("cluster_b") >= 0)
        )

        if pairs.is_empty():
            return pl.DataFrame(
                schema={"cluster_a": pl.Int64, "cluster_b": pl.Int64, "mean_xgf_pct": pl.Float64}
            )

        return (
            pairs
            .group_by(["cluster_a", "cluster_b"])
            .agg(pl.col("model_xgf_pct").mean().alias("mean_xgf_pct"))
            .sort(["cluster_a", "cluster_b"])
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if self._kmeans is None:
            raise RuntimeError("Model not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "kmeans":              self._kmeans,
            "scaler":              self._scaler,
            "k":                   self._k,
            "feature_names":       self._feature_names,
            "archetype_names":     self._archetype_names,
            "custom_names":        self._custom_names,
            "centroids_original":  self._centroids_original,
            "cluster_sizes":       self._cluster_sizes,
            "version":             self._version,
            "random_state":        self._random_state,
            "n_init":              self._n_init,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PlayerArchetypeModel":
        payload = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._kmeans               = payload["kmeans"]
        obj._scaler               = payload["scaler"]
        obj._k                    = payload["k"]
        obj._feature_names        = payload.get("feature_names", list(SKATER_FEATURE_NAMES))
        obj._archetype_names      = payload.get("archetype_names", [])
        obj._custom_names         = payload.get("custom_names", {})
        obj._centroids_original   = payload.get("centroids_original")
        obj._cluster_sizes        = payload.get("cluster_sizes")
        obj._version              = payload.get("version", MODEL_VERSION)
        obj._random_state         = payload.get("random_state", 42)
        obj._n_init               = payload.get("n_init", 20)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_archetypes(
    assignments_df: pl.DataFrame,
    centroids_df:   pl.DataFrame,
    output_dir: Path,
    season: int,
) -> tuple[Path, Path]:
    """Write archetype outputs to parquets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_assign  = output_dir / f"player_archetypes_{season}.parquet"
    p_centers = output_dir / f"archetype_centroids_{season}.parquet"

    for col, dtype in PLAYER_ARCHETYPE_SCHEMA.items():
        if col not in assignments_df.columns:
            assignments_df = assignments_df.with_columns(pl.lit(None).cast(dtype).alias(col))
    assignments_df.select(list(PLAYER_ARCHETYPE_SCHEMA.keys())).write_parquet(p_assign)
    centroids_df.write_parquet(p_centers)

    return p_assign, p_centers


def read_archetypes(data_dir: Path, season: int) -> pl.DataFrame:
    """Read player archetype assignments."""
    path = Path(data_dir) / "archetypes" / f"player_archetypes_{season}.parquet"
    return pl.read_parquet(path)
