"""Model Versioning — Feature 2.8.

Snapshot registry for reproducible backtest comparisons.  Stores metadata
for every model file saved during a training run, enabling exact replay of
any historical model state.

Each registered snapshot has:
    - Unique slug:  ``{model_type}_{season}_{date}``
    - Path to the .pkl file on disk
    - Training metrics (RMSE, AUC, etc.)
    - Feature list (for drift detection)
    - Timestamp

The registry itself is stored as a JSON file at
``~/.gretzky/models/registry.json`` (or a caller-specified path).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# SnapshotMetadata
# ---------------------------------------------------------------------------

@dataclass
class SnapshotMetadata:
    """Immutable record for one model snapshot.

    Attributes:
        slug:          Unique identifier, e.g. ``xg_2025_2026-03-28``.
        model_type:    Short model name, e.g. ``xg``, ``rapm``, ``goalie``.
        season:        NHL season start-year, e.g. 2025.
        created_at:    ISO-8601 UTC timestamp of when fit() was called.
        path:          Absolute path to the serialised .pkl file.
        metrics:       Dict of training/eval metrics (RMSE, AUC, etc.).
        feature_names: Feature names used at training time.
        params:        Hyperparameters dict.
        notes:         Free-text notes (e.g. "retrained after data fix").
    """
    slug:          str
    model_type:    str
    season:        int
    created_at:    str                   # ISO-8601 UTC
    path:          str                   # absolute path str
    metrics:       dict[str, float]      = field(default_factory=dict)
    feature_names: list[str]             = field(default_factory=list)
    params:        dict[str, Any]        = field(default_factory=dict)
    notes:         str                   = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SnapshotMetadata":
        return cls(
            slug          = d["slug"],
            model_type    = d["model_type"],
            season        = int(d["season"]),
            created_at    = d["created_at"],
            path          = d["path"],
            metrics       = d.get("metrics", {}),
            feature_names = d.get("feature_names", []),
            params        = d.get("params", {}),
            notes         = d.get("notes", ""),
        )

    def exists(self) -> bool:
        """Return True if the .pkl file still exists on disk."""
        return Path(self.path).exists()


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Persistent registry of model snapshots.

    Usage::

        registry = ModelRegistry()

        # After fitting a model:
        meta = registry.register(
            model_type="goalie",
            season=2025,
            path=Path("~/.gretzky/models/goalie_model.pkl"),
            metrics={"rmse": 0.32, "n_goalies": 180},
            feature_names=["sv_pct", "xga_per60", ...],
            params={"n_estimators": 300},
        )

        # Look up later:
        latest = registry.latest("goalie", 2025)
        all_snaps = registry.list(model_type="goalie")
        snap = registry.get("goalie_2025_2026-03-28")
        registry.delete("goalie_2025_old")
    """

    _DEFAULT_PATH = Path.home() / ".gretzky" / "models" / "registry.json"

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = Path(registry_path) if registry_path else self._DEFAULT_PATH
        self._snapshots: dict[str, SnapshotMetadata] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load registry from disk (silently starts empty if not found)."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._snapshots = {
                    slug: SnapshotMetadata.from_dict(d)
                    for slug, d in raw.get("snapshots", {}).items()
                }
            except (json.JSONDecodeError, KeyError):
                self._snapshots = {}

    def _save(self) -> None:
        """Persist registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshots": {
                slug: meta.to_dict()
                for slug, meta in self._snapshots.items()
            }
        }
        self._path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model_type: str,
        season: int,
        path: Path,
        metrics:       dict[str, float] | None = None,
        feature_names: list[str]        | None = None,
        params:        dict[str, Any]   | None = None,
        notes:         str = "",
        slug:          str | None = None,
    ) -> SnapshotMetadata:
        """Register a new model snapshot and persist registry.

        Args:
            model_type:    Short identifier, e.g. ``xg``, ``rapm``.
            season:        NHL season start-year.
            path:          Path to the serialised model file.
            metrics:       Training/eval metrics dict.
            feature_names: Feature names used at training time.
            params:        Hyperparameter dict.
            notes:         Free-text description.
            slug:          Override auto-generated slug.

        Returns:
            SnapshotMetadata for the registered snapshot.
        """
        path = Path(path)
        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")

        if slug is None:
            base_slug = f"{model_type}_{season}_{date_str}"
            slug = base_slug
            # Append counter if slug already used today
            counter = 1
            while slug in self._snapshots:
                slug = f"{base_slug}_{counter}"
                counter += 1

        meta = SnapshotMetadata(
            slug          = slug,
            model_type    = model_type,
            season        = season,
            created_at    = now_utc.isoformat(),
            path          = str(path.expanduser().resolve()),
            metrics       = metrics or {},
            feature_names = feature_names or [],
            params        = params or {},
            notes         = notes,
        )
        self._snapshots[slug] = meta
        self._save()
        return meta

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, slug: str) -> SnapshotMetadata | None:
        """Return snapshot by slug, or None if not found."""
        return self._snapshots.get(slug)

    def list(
        self,
        model_type: str | None = None,
        season:     int | None = None,
    ) -> list[SnapshotMetadata]:
        """Return snapshots filtered by model_type and/or season.

        Results are sorted by created_at descending (newest first).
        """
        snaps = list(self._snapshots.values())
        if model_type is not None:
            snaps = [s for s in snaps if s.model_type == model_type]
        if season is not None:
            snaps = [s for s in snaps if s.season == season]
        return sorted(snaps, key=lambda s: s.created_at, reverse=True)

    def latest(
        self,
        model_type: str,
        season:     int,
    ) -> SnapshotMetadata | None:
        """Return the most recently registered snapshot for (model_type, season)."""
        candidates = self.list(model_type=model_type, season=season)
        return candidates[0] if candidates else None

    def compare(
        self,
        slug_a: str,
        slug_b: str,
        metric: str = "rmse",
    ) -> dict[str, Any]:
        """Compare two snapshots on a given metric.

        Returns:
            {
                "slug_a": slug_a,
                "slug_b": slug_b,
                "metric": metric,
                "value_a": float or None,
                "value_b": float or None,
                "delta": value_b - value_a or None,
                "better": slug_a / slug_b / "tie" / None,
            }

        'Better' is defined as lower for loss metrics (rmse, brier) and
        higher for score metrics (auc, r2, correlation).
        """
        a = self.get(slug_a)
        b = self.get(slug_b)
        val_a = a.metrics.get(metric) if a else None
        val_b = b.metrics.get(metric) if b else None

        delta = None
        better = None
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            lower_is_better = metric.lower() in {"rmse", "mae", "brier", "brier_score", "loss"}
            if lower_is_better:
                if val_a < val_b:
                    better = slug_a
                elif val_b < val_a:
                    better = slug_b
                else:
                    better = "tie"
            else:
                if val_a > val_b:
                    better = slug_a
                elif val_b > val_a:
                    better = slug_b
                else:
                    better = "tie"

        return {
            "slug_a":  slug_a,
            "slug_b":  slug_b,
            "metric":  metric,
            "value_a": val_a,
            "value_b": val_b,
            "delta":   delta,
            "better":  better,
        }

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, slug: str, delete_file: bool = False) -> bool:
        """Remove snapshot from registry.

        Args:
            slug:        Snapshot slug to remove.
            delete_file: If True, also delete the .pkl file from disk.

        Returns:
            True if the slug existed and was removed, False otherwise.
        """
        meta = self._snapshots.pop(slug, None)
        if meta is None:
            return False
        if delete_file:
            p = Path(meta.path)
            if p.exists():
                p.unlink()
        self._save()
        return True

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the registry."""
        snaps = self.list()
        if not snaps:
            return "Registry is empty."
        lines = [f"ModelRegistry ({len(snaps)} snapshots):"]
        for s in snaps:
            exists_flag = "✓" if s.exists() else "✗ missing"
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
            lines.append(
                f"  {s.slug:<35s}  season={s.season}  [{exists_flag}]  {metrics_str}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __repr__(self) -> str:
        return f"ModelRegistry(path={self._path}, n={len(self)})"
