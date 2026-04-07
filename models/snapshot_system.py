"""Model Snapshot System — Feature 2.16.

Full GRTZKY system state snapshot: serialises all running model states
into one ``GRETZKY_as_of_{date}.pkl`` file for reproducible replay and
historical comparison.

Unlike Feature 2.8 (ModelRegistry) which tracks individual model file
paths, this captures the complete in-memory runtime state:
    - InSeasonBayesianPipeline (updater + historical priors)
    - Latest EWMA form DataFrame
    - RegimeChangeDetector state + active alerts
    - StrengthOfScheduleNormalizer opponent quality map
    - Metadata: date, season, game_count, notes

Restore any snapshot to replay exactly what the system believed at
that point in the season — critical for backtesting Polymarket edge
detection against what was actually known at bet time.

Usage::

    from models.snapshot_system import GRETZKYSnapshot

    # Save current state:
    snap = GRETZKYSnapshot.capture(
        pipeline=pipeline,
        ewma_df=form_summary,
        regime_detector=detector,
        sos_normalizer=normalizer,
        season=2025,
        game_count=42,
        notes="Pre-trade deadline state",
    )
    snap.save(Path("~/.gretzky/snapshots/"))

    # Load and compare:
    snap_a = GRETZKYSnapshot.load(path_a)
    snap_b = GRETZKYSnapshot.load(path_b)
    diff = GRETZKYSnapshot.compare(snap_a, snap_b)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNAPSHOT_VERSION = "gretzky_snap_v1"
DEFAULT_SNAPSHOT_DIR = Path.home() / ".gretzky" / "snapshots"


# ---------------------------------------------------------------------------
# GRETZKYSnapshot
# ---------------------------------------------------------------------------

class GRETZKYSnapshot:
    """Complete serialised GRTZKY system state at a point in time.

    Attributes:
        name:          Human-readable snapshot name (auto-generated if not given).
        created_at:    ISO-8601 UTC timestamp.
        season:        NHL season start-year.
        game_count:    Number of games processed this season.
        notes:         Free-text description (Bob-provided context).
        pipeline:      InSeasonBayesianPipeline state.
        ewma_df:       Latest EWMA form summary DataFrame (or None).
        regime_detector: RegimeChangeDetector state (or None).
        sos_normalizer:  StrengthOfScheduleNormalizer state (or None).
        extra:         Any additional key-value data.
    """

    def __init__(
        self,
        name:             str,
        created_at:       str,
        season:           int,
        game_count:       int,
        notes:            str,
        pipeline:         Any,
        ewma_df:          pl.DataFrame | None,
        regime_detector:  Any,
        sos_normalizer:   Any,
        extra:            dict[str, Any] | None = None,
    ) -> None:
        self.name            = name
        self.created_at      = created_at
        self.season          = season
        self.game_count      = game_count
        self.notes           = notes
        self.pipeline        = pipeline
        self.ewma_df         = ewma_df
        self.regime_detector = regime_detector
        self.sos_normalizer  = sos_normalizer
        self.extra           = extra or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def capture(
        cls,
        pipeline:         Any,
        season:           int,
        game_count:       int = 0,
        ewma_df:          pl.DataFrame | None = None,
        regime_detector:  Any = None,
        sos_normalizer:   Any = None,
        name:             str | None = None,
        notes:            str = "",
        extra:            dict[str, Any] | None = None,
    ) -> "GRETZKYSnapshot":
        """Capture current system state into a snapshot object.

        Args:
            pipeline:        InSeasonBayesianPipeline instance.
            season:          NHL season start-year.
            game_count:      Number of games processed this season.
            ewma_df:         Latest EWMA form summary DataFrame.
            regime_detector: RegimeChangeDetector instance.
            sos_normalizer:  StrengthOfScheduleNormalizer instance.
            name:            Snapshot name (auto-generated: GRETZKY_as_of_{date}).
            notes:           Bob's free-text context.
            extra:           Any extra key-value data to persist.

        Returns:
            GRETZKYSnapshot ready to save.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        if name is None:
            name = f"GRETZKY_as_of_{date_str}"

        return cls(
            name            = name,
            created_at      = now.isoformat(),
            season          = season,
            game_count      = game_count,
            notes           = notes,
            pipeline        = pipeline,
            ewma_df         = ewma_df,
            regime_detector = regime_detector,
            sos_normalizer  = sos_normalizer,
            extra           = extra or {},
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, snapshot_dir: Path | None = None) -> Path:
        """Serialise snapshot to disk.

        Args:
            snapshot_dir: Directory to write to (default: ~/.gretzky/snapshots/).

        Returns:
            Path to the saved .pkl file.
        """
        snap_dir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Sanitise name for use as filename
        safe_name = self.name.replace(" ", "_").replace("/", "_")
        path = snap_dir / f"{safe_name}.pkl"

        payload = {
            "version":        SNAPSHOT_VERSION,
            "name":           self.name,
            "created_at":     self.created_at,
            "season":         self.season,
            "game_count":     self.game_count,
            "notes":          self.notes,
            "pipeline":       self.pipeline,
            "ewma_df":        self.ewma_df,
            "regime_detector": self.regime_detector,
            "sos_normalizer": self.sos_normalizer,
            "extra":          self.extra,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "GRETZKYSnapshot":
        """Load a snapshot from disk."""
        payload = joblib.load(Path(path))
        return cls(
            name            = payload["name"],
            created_at      = payload["created_at"],
            season          = payload["season"],
            game_count      = payload.get("game_count", 0),
            notes           = payload.get("notes", ""),
            pipeline        = payload.get("pipeline"),
            ewma_df         = payload.get("ewma_df"),
            regime_detector = payload.get("regime_detector"),
            sos_normalizer  = payload.get("sos_normalizer"),
            extra           = payload.get("extra", {}),
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        snap_a: "GRETZKYSnapshot",
        snap_b: "GRETZKYSnapshot",
    ) -> dict[str, Any]:
        """Compare two snapshots and summarise key differences.

        Returns:
            Dict with:
                name_a, name_b, created_at_a, created_at_b,
                game_count_delta, n_players_a, n_players_b,
                n_active_regimes_a, n_active_regimes_b,
                ewma_mean_xgf60_a, ewma_mean_xgf60_b.
        """
        def _n_players(snap: GRETZKYSnapshot) -> int:
            if snap.pipeline is None:
                return 0
            try:
                return snap.pipeline.n_players_observed()
            except Exception:
                return 0

        def _n_regimes(snap: GRETZKYSnapshot) -> int:
            if snap.regime_detector is None:
                return 0
            try:
                return len(snap.regime_detector.active_regimes())
            except Exception:
                return 0

        def _mean_ewma(snap: GRETZKYSnapshot) -> float | None:
            if snap.ewma_df is None or snap.ewma_df.is_empty():
                return None
            try:
                return float(snap.ewma_df["ewma_xgf60"].mean())
            except Exception:
                return None

        return {
            "name_a":               snap_a.name,
            "name_b":               snap_b.name,
            "created_at_a":         snap_a.created_at,
            "created_at_b":         snap_b.created_at,
            "season_a":             snap_a.season,
            "season_b":             snap_b.season,
            "game_count_a":         snap_a.game_count,
            "game_count_b":         snap_b.game_count,
            "game_count_delta":     snap_b.game_count - snap_a.game_count,
            "n_players_a":          _n_players(snap_a),
            "n_players_b":          _n_players(snap_b),
            "n_active_regimes_a":   _n_regimes(snap_a),
            "n_active_regimes_b":   _n_regimes(snap_b),
            "ewma_mean_xgf60_a":    _mean_ewma(snap_a),
            "ewma_mean_xgf60_b":    _mean_ewma(snap_b),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return human-readable snapshot summary."""
        n_players = 0
        if self.pipeline is not None:
            try:
                n_players = self.pipeline.n_players_observed()
            except Exception:
                pass

        n_regimes = 0
        if self.regime_detector is not None:
            try:
                n_regimes = len(self.regime_detector.active_regimes())
            except Exception:
                pass

        ewma_rows = len(self.ewma_df) if self.ewma_df is not None else 0

        lines = [
            f"GRETZKYSnapshot: {self.name}",
            f"  Created:    {self.created_at}",
            f"  Season:     {self.season}",
            f"  Game count: {self.game_count}",
            f"  Players:    {n_players}",
            f"  EWMA rows:  {ewma_rows}",
            f"  Regimes:    {n_regimes} active",
        ]
        if self.notes:
            lines.append(f"  Notes:      {self.notes}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GRETZKYSnapshot(name='{self.name}', season={self.season}, game={self.game_count})"


# ---------------------------------------------------------------------------
# SnapshotIndex — list and manage snapshots on disk
# ---------------------------------------------------------------------------

class SnapshotIndex:
    """Manages a directory of GRTZKY snapshots.

    Usage::

        index = SnapshotIndex()
        index.list()           # → list of (name, path, created_at)
        snap = index.load("GRETZKY_as_of_2026-03-28")
        index.delete("GRETZKY_as_of_2026-03-01")
    """

    def __init__(self, snapshot_dir: Path | None = None) -> None:
        self._dir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, str]]:
        """Return sorted list of available snapshots (newest first).

        Returns:
            List of {"name": str, "path": str, "created_at": str}.
        """
        entries: list[dict[str, str]] = []
        for p in sorted(self._dir.glob("*.pkl")):
            try:
                # Load just the metadata without deserialising the full payload
                payload = joblib.load(p)
                entries.append({
                    "name":       payload.get("name", p.stem),
                    "path":       str(p),
                    "created_at": payload.get("created_at", ""),
                    "season":     str(payload.get("season", "")),
                    "game_count": str(payload.get("game_count", "")),
                    "notes":      payload.get("notes", ""),
                })
            except Exception:
                entries.append({"name": p.stem, "path": str(p), "created_at": ""})

        return sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)

    def load(self, name: str) -> GRETZKYSnapshot:
        """Load a snapshot by name (without .pkl extension)."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        path = self._dir / f"{safe_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")
        return GRETZKYSnapshot.load(path)

    def delete(self, name: str) -> bool:
        """Delete a snapshot by name.  Returns True if deleted."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        path = self._dir / f"{safe_name}.pkl"
        if path.exists():
            path.unlink()
            return True
        return False

    def save(self, snap: GRETZKYSnapshot) -> Path:
        """Save a snapshot to this index's directory."""
        return snap.save(self._dir)

    def __len__(self) -> int:
        return len(list(self._dir.glob("*.pkl")))

    def __repr__(self) -> str:
        return f"SnapshotIndex(dir={self._dir}, n={len(self)})"
