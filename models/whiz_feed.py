"""The Whiz's game-observation feed — bridges game-watching intelligence to simulation.

The Whiz watches games with Bob and writes observations about players to
``data/whiz_brain/observations.json``.  This module reads those observations
and converts them into per-player simulation adjustments that extend the
hot-hand signal (Feature 2.28) with real human-in-the-loop signal.

Each observation carries:
  - ``subject_name`` / ``player_id`` / ``team``
  - ``observation``  — plain-English text (what The Whiz saw)
  - ``category``     — form | tendency | injury_concern | chemistry | coaching | goalie
  - ``confidence``   — [0.0, 1.0]  (The Whiz's certainty)
  - ``sim_delta``    — dict of simulation adjustments (hot_hand_score, xg_multiplier, …)
  - ``expires_after_games`` — how many games before this observation fades (default 5)

Usage::

    from models.whiz_feed import WhizFeed

    feed = WhizFeed()
    adjustments = feed.get_player_adjustments()
    # → {player_id: {"hot_hand_score": 1.2, "xg_multiplier": 1.06}}

    # When building pre-game inputs for Rust engine:
    for player in lineup:
        adj = adjustments.get(player.player_id, {})
        hot_hand = adj.get("hot_hand_score", 0.0)
        xg_mult  = adj.get("xg_multiplier", 1.0)
        # Apply to player feature vector before passing to gretzky_engine

Writing observations (done by The Whiz during game sessions)::

    feed.add_observation(
        subject_name="Cole Caufield",
        player_id="8481600",
        team="MTL",
        opponent="TBL",
        game_id="2024021000",
        game_date="2026-03-28",
        observation="Getting to dangerous spots — 3 grade-A chances in first 2 periods. Puck is finding him on the PP. Peak form.",
        category="form",
        confidence=0.8,
        sim_delta={"hot_hand_score": 1.2, "xg_multiplier": 1.08},
        expires_after_games=5,
    )
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_REPO = Path(__file__).parents[1]
OBSERVATIONS_PATH = _REPO / "data" / "whiz_brain" / "observations.json"

Category = Literal["form", "tendency", "injury_concern", "chemistry", "coaching", "goalie"]


class WhizFeed:
    """Read/write The Whiz's game observations and export simulation adjustments."""

    def __init__(self, path: Path = OBSERVATIONS_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def load(self) -> dict:
        if not self._path.exists():
            return {"observations": []}
        with open(self._path) as fh:
            return json.load(fh)

    def all_observations(self) -> list[dict]:
        return self.load().get("observations", [])

    def active_observations(self, as_of_date: str | None = None) -> list[dict]:
        """Return observations where ``active == True``."""
        return [o for o in self.all_observations() if o.get("active", True)]

    def get_player_adjustments(self) -> dict[str, dict]:
        """Return ``{player_id: sim_delta_dict}`` for all active observations.

        When a player has multiple active observations, ``hot_hand_score`` values
        are summed (capped at ±2.0) and ``xg_multiplier`` values are multiplied
        (capped at [0.75, 1.30]) to avoid runaway adjustments.
        """
        active = self.active_observations()
        result: dict[str, dict] = {}

        for obs in active:
            pid   = str(obs.get("player_id", "")).strip()
            delta = obs.get("sim_delta", {})
            if not pid or not delta:
                continue

            if pid not in result:
                result[pid] = {"hot_hand_score": 0.0, "xg_multiplier": 1.0}

            conf = float(obs.get("confidence", 0.5))

            if "hot_hand_score" in delta:
                result[pid]["hot_hand_score"] += float(delta["hot_hand_score"]) * conf

            if "xg_multiplier" in delta:
                # Blend toward the observed multiplier weighted by confidence
                m = float(delta["xg_multiplier"])
                result[pid]["xg_multiplier"] *= 1.0 + (m - 1.0) * conf

        # Cap values
        for pid, adj in result.items():
            adj["hot_hand_score"] = max(-2.0, min(2.0, adj["hot_hand_score"]))
            adj["xg_multiplier"]  = max(0.75, min(1.30, adj["xg_multiplier"]))

        return result

    # ------------------------------------------------------------------
    # Write API  (used by The Whiz during game sessions)
    # ------------------------------------------------------------------

    def add_observation(
        self,
        subject_name: str,
        team: str,
        observation: str,
        category: Category = "form",
        confidence: float = 0.6,
        sim_delta: dict | None = None,
        player_id: str = "",
        opponent: str = "",
        game_id: str = "",
        game_date: str = "",
        expires_after_games: int = 5,
        subject_type: str = "player",
        source: str = "manual",
    ) -> str:
        """Append a new observation. Returns the assigned observation id.

        Args:
            source: ``"manual"`` for Bob's hand-typed observations;
                    ``"ai_observer"`` for Feature 16.13 auto-generated ones.
        """
        data = self.load()
        obs_id = str(uuid.uuid4())[:8]
        entry = {
            "id":                  obs_id,
            "subject_type":        subject_type,
            "subject_name":        subject_name,
            "player_id":           player_id,
            "team":                team,
            "opponent":            opponent,
            "game_id":             game_id,
            "game_date":           game_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "observation":         observation,
            "category":            category,
            "confidence":          round(float(confidence), 2),
            "sim_delta":           sim_delta or {},
            "expires_after_games": expires_after_games,
            "active":              True,
            "source":              source,
            "noted_at":            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        data["observations"].append(entry)
        self._save(data)
        return obs_id

    def deactivate(self, obs_id: str) -> bool:
        """Mark observation as inactive (expired/overridden). Returns True if found."""
        data = self.load()
        for obs in data["observations"]:
            if obs.get("id") == obs_id:
                obs["active"] = False
                self._save(data)
                return True
        return False

    def deactivate_player(self, player_id: str) -> int:
        """Deactivate all observations for a player. Returns count deactivated."""
        data  = self.load()
        count = 0
        for obs in data["observations"]:
            if str(obs.get("player_id", "")) == str(player_id) and obs.get("active", True):
                obs["active"] = False
                count += 1
        if count:
            self._save(data)
        return count

    def clear_all(self) -> None:
        """Wipe all observations (use with care)."""
        self._save({"observations": []})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self, data: dict) -> None:
        with open(self._path, "w") as fh:
            json.dump(data, fh, indent=2)
        # Keep newest first in file for readability
        data["observations"].sort(key=lambda o: o.get("noted_at", ""), reverse=True)
        with open(self._path, "w") as fh:
            json.dump(data, fh, indent=2)
