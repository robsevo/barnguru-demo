"""In-Season Bayesian Posterior Pipeline — Feature 2.12.

Sequential game-by-game posterior update with automatic weight blending:
  - Prior: historical posterior from Feature 2.9 (BayesianPlayerRating)
  - Current: SequentialBayesianUpdater state from Feature 2.10
  - Blend weight w(g) shifts from 25% current at Game 10 → 73% current at Game 82

Blend formula (linear interpolation):
    w(g) = clip(0.25 + 0.48 × (g − 10) / 72,  min=0.0,  max=0.73)

Blended rating:
    μ_blend    = w × μ_current  +  (1−w) × μ_historical
    σ_blend    = sqrt(w² × σ_current²  +  (1−w)² × σ_historical²)
    CI_95      = μ_blend ± 1.960 × σ_blend

Outputs
-------
``inseason_ratings_{season}_game{N}.parquet`` — snapshot after game N:
    player_id, player_name, season, games_played,
    blend_weight, mu_blend, sigma_blend, ci_lower_95, ci_upper_95,
    mu_historical, mu_current, model_version
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning
from models.bayesian_player_rating import (
    SequentialBayesianUpdater,
    LEAGUE_MEAN_XGF60,
    LEAGUE_SIGMA_XGF60,
    OBS_SIGMA_XGF60,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION      = "inseason_bayes_v1"
BLEND_W_GAME10     = 0.25   # weight on current-season at game 10
BLEND_W_GAME82     = 0.73   # weight on current-season at game 82
BLEND_GAME_LOW     = 10     # game count where BLEND_W_GAME10 applies
BLEND_GAME_HIGH    = 82     # game count where BLEND_W_GAME82 applies
Z95                = 1.959964


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

INSEASON_RATING_SCHEMA: dict[str, pl.DataType] = {
    "player_id":      pl.Int64,
    "player_name":    pl.Utf8,
    "season":         pl.Int64,
    "games_played":   pl.Int64,
    "blend_weight":   pl.Float64,
    "mu_blend":       pl.Float64,
    "sigma_blend":    pl.Float64,
    "ci_lower_95":    pl.Float64,
    "ci_upper_95":    pl.Float64,
    "mu_historical":  pl.Float64,
    "sigma_historical": pl.Float64,
    "mu_current":     pl.Float64,
    "sigma_current":  pl.Float64,
    "model_version":  pl.Utf8,
}


# ---------------------------------------------------------------------------
# Blend weight function
# ---------------------------------------------------------------------------

def blend_weight(games_played: int | float) -> float:
    """Return blend weight w(g) ∈ [0.0, 0.73] for a given game count.

    w(g) = clip(0.25 + 0.48 × (g − 10) / 72,  0.0,  0.73)

    At g=10:  w = 0.25  (75% historical, 25% current)
    At g=82:  w = 0.73  (27% historical, 73% current)
    At g<10:  w < 0.25  (even more weight on historical prior)
    At g>82:  w = 0.73  (capped)
    """
    g = float(games_played)
    raw = BLEND_W_GAME10 + (BLEND_W_GAME82 - BLEND_W_GAME10) * (g - BLEND_GAME_LOW) / (
        BLEND_GAME_HIGH - BLEND_GAME_LOW
    )
    return float(np.clip(raw, 0.0, BLEND_W_GAME82))


def blend_weights_array(games_played: np.ndarray) -> np.ndarray:
    """Vectorised blend_weight for an array of game counts."""
    raw = BLEND_W_GAME10 + (BLEND_W_GAME82 - BLEND_W_GAME10) * (
        games_played.astype(float) - BLEND_GAME_LOW
    ) / (BLEND_GAME_HIGH - BLEND_GAME_LOW)
    return np.clip(raw, 0.0, BLEND_W_GAME82)


# ---------------------------------------------------------------------------
# Blending logic
# ---------------------------------------------------------------------------

def blend_posteriors(
    mu_hist: float,
    sigma_hist: float,
    mu_curr: float,
    sigma_curr: float,
    games_played: int,
) -> tuple[float, float, float]:
    """Blend historical and current-season posteriors.

    Args:
        mu_hist:      Historical prior mean (from Feature 2.9).
        sigma_hist:   Historical prior sigma.
        mu_curr:      Current-season posterior mean (from Feature 2.10).
        sigma_curr:   Current-season posterior sigma.
        games_played: Games played so far this season.

    Returns:
        (mu_blend, sigma_blend, blend_weight_w)
    """
    w = blend_weight(games_played)
    mu_blend    = w * mu_curr + (1.0 - w) * mu_hist
    sigma_blend = float(np.sqrt(w**2 * sigma_curr**2 + (1.0 - w)**2 * sigma_hist**2))
    return mu_blend, sigma_blend, w


# ---------------------------------------------------------------------------
# InSeasonBayesianPipeline — Feature 2.12
# ---------------------------------------------------------------------------

class InSeasonBayesianPipeline:
    """In-season Bayesian posterior pipeline (Feature 2.12).

    Maintains a SequentialBayesianUpdater internally and blends its
    per-game updates with the historical prior (Feature 2.9 output)
    using the weight schedule w(g).

    Usage::

        pipeline = InSeasonBayesianPipeline(historical_ratings_df)

        # After each game:
        pipeline.observe(player_id=8478402, player_name="Matthews", xgf_per60=3.4)

        # Get current blended ratings for all players:
        ratings = pipeline.blended_ratings(season=2025)
    """

    def __init__(
        self,
        historical_ratings_df: pl.DataFrame | None = None,
        obs_sigma: float = OBS_SIGMA_XGF60,
        league_mean:  float = LEAGUE_MEAN_XGF60,
        league_sigma: float = LEAGUE_SIGMA_XGF60,
    ) -> None:
        """
        Args:
            historical_ratings_df: Feature 2.9 output — must have columns
                player_id, posterior_mean, posterior_sigma (and optionally
                player_name).  None = use flat league-mean prior for everyone.
            obs_sigma:   Per-game observation noise.
            league_mean: Fallback prior mean for players not in historical_df.
            league_sigma: Fallback prior sigma.
        """
        self._obs_sigma    = obs_sigma
        self._league_mean  = league_mean
        self._league_sigma = league_sigma
        self._version      = MODEL_VERSION

        # Build lookup: player_id → (hist_mu, hist_sigma, player_name)
        self._historical: dict[int, tuple[float, float, str]] = {}
        if historical_ratings_df is not None:
            self._load_historical(historical_ratings_df)

        self._updater = SequentialBayesianUpdater(
            league_mean  = league_mean,
            league_sigma = league_sigma,
            obs_sigma    = obs_sigma,
        )

    def _load_historical(self, df: pl.DataFrame) -> None:
        required = {"player_id", "posterior_mean", "posterior_sigma"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"historical_ratings_df missing columns: {sorted(missing)}")
        for row in df.to_dicts():
            pid    = int(row["player_id"])
            mu     = float(row["posterior_mean"])
            sigma  = float(row["posterior_sigma"])
            name   = str(row.get("player_name") or "")
            self._historical[pid] = (mu, sigma, name)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        player_id:   int,
        player_name: str,
        xgf_per60:   float,
        obs_sigma:   float | None = None,
    ) -> None:
        """Record one game observation for a player.

        Args:
            player_id:   NHL player ID.
            player_name: Display name.
            xgf_per60:   Observed xGF/60 in this game.
            obs_sigma:   Override per-game noise (default: self._obs_sigma).
        """
        self._updater.update(
            player_id   = player_id,
            player_name = player_name,
            observation = xgf_per60,
            obs_sigma   = obs_sigma,
        )

    def observe_game(
        self,
        game_df: pl.DataFrame,
        xgf_col: str = "xgf_per60",
    ) -> None:
        """Record all player observations from a single game DataFrame.

        Args:
            game_df: One row per player with columns player_id, player_name,
                     and xgf_col (observation value).
            xgf_col: Column name for the observation metric.
        """
        required = {"player_id", xgf_col}
        missing = required - set(game_df.columns)
        if missing:
            raise ValueError(f"game_df missing columns: {sorted(missing)}")

        for row in game_df.to_dicts():
            pid  = int(row["player_id"])
            name = str(row.get("player_name") or "")
            val  = row.get(xgf_col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            self.observe(pid, name, float(val))

    # ------------------------------------------------------------------
    # Blended ratings
    # ------------------------------------------------------------------

    def player_blended_rating(
        self,
        player_id: int,
        season: int,
    ) -> dict[str, float | int | str] | None:
        """Return blended rating dict for one player, or None if unknown."""
        state = self._updater.get(player_id)
        if state is None:
            return None

        hist_mu, hist_sigma, name = self._historical.get(
            player_id,
            (self._league_mean, self._league_sigma, ""),
        )

        mu_blend, sigma_blend, w = blend_posteriors(
            mu_hist   = hist_mu,
            sigma_hist = hist_sigma,
            mu_curr   = state.mu,
            sigma_curr = state.sigma,
            games_played = state.n_games,
        )

        return {
            "player_id":        player_id,
            "player_name":      state.player_name or name,
            "season":           season,
            "games_played":     state.n_games,
            "blend_weight":     w,
            "mu_blend":         mu_blend,
            "sigma_blend":      sigma_blend,
            "ci_lower_95":      mu_blend - Z95 * sigma_blend,
            "ci_upper_95":      mu_blend + Z95 * sigma_blend,
            "mu_historical":    hist_mu,
            "sigma_historical": hist_sigma,
            "mu_current":       state.mu,
            "sigma_current":    state.sigma,
            "model_version":    self._version,
        }

    def blended_ratings(self, season: int) -> pl.DataFrame:
        """Return blended ratings for all observed players as a DataFrame.

        Returns:
            DataFrame matching INSEASON_RATING_SCHEMA, sorted by mu_blend desc.
        """
        states = self._updater.all_states()
        if not states:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in INSEASON_RATING_SCHEMA.items()}
            )

        rows: dict[str, list] = {col: [] for col in INSEASON_RATING_SCHEMA}

        for state in states:
            hist_mu, hist_sigma, hist_name = self._historical.get(
                state.player_id,
                (self._league_mean, self._league_sigma, ""),
            )
            mu_blend, sigma_blend, w = blend_posteriors(
                mu_hist    = hist_mu,
                sigma_hist = hist_sigma,
                mu_curr    = state.mu,
                sigma_curr = state.sigma,
                games_played = state.n_games,
            )

            rows["player_id"].append(state.player_id)
            rows["player_name"].append(state.player_name or hist_name)
            rows["season"].append(season)
            rows["games_played"].append(state.n_games)
            rows["blend_weight"].append(w)
            rows["mu_blend"].append(mu_blend)
            rows["sigma_blend"].append(sigma_blend)
            rows["ci_lower_95"].append(mu_blend - Z95 * sigma_blend)
            rows["ci_upper_95"].append(mu_blend + Z95 * sigma_blend)
            rows["mu_historical"].append(hist_mu)
            rows["sigma_historical"].append(hist_sigma)
            rows["mu_current"].append(state.mu)
            rows["sigma_current"].append(state.sigma)
            rows["model_version"].append(self._version)

        df = pl.DataFrame(rows).cast(INSEASON_RATING_SCHEMA)
        return df.sort("mu_blend", descending=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def n_players_observed(self) -> int:
        """Number of players with at least one game observation."""
        return len(self._updater.all_states())

    def games_played(self, player_id: int) -> int:
        """Return games observed for a player (0 if unknown)."""
        state = self._updater.get(player_id)
        return state.n_games if state else 0

    def reset_player(self, player_id: int) -> None:
        """Reset a player's current-season history (keeps historical prior)."""
        self._updater.reset(player_id)

    def reset_all(self) -> None:
        """Reset all current-season history."""
        self._updater.reset()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise pipeline state (historical priors + updater state)."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "historical":  self._historical,
            "updater":     self._updater,
            "obs_sigma":   self._obs_sigma,
            "league_mean": self._league_mean,
            "league_sigma": self._league_sigma,
            "version":     self._version,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "InSeasonBayesianPipeline":
        import joblib
        payload = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._historical   = payload["historical"]
        obj._updater      = payload["updater"]
        obj._obs_sigma    = payload.get("obs_sigma",    OBS_SIGMA_XGF60)
        obj._league_mean  = payload.get("league_mean",  LEAGUE_MEAN_XGF60)
        obj._league_sigma = payload.get("league_sigma", LEAGUE_SIGMA_XGF60)
        obj._version      = payload.get("version",      MODEL_VERSION)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_inseason_ratings(
    df: pl.DataFrame,
    output_dir: Path,
    season: int,
    game_number: int | None = None,
) -> Path:
    """Write in-season ratings snapshot to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_game{game_number}" if game_number is not None else ""
    path = output_dir / f"inseason_ratings_{season}{suffix}.parquet"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    for col, dtype in INSEASON_RATING_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(INSEASON_RATING_SCHEMA.keys())).write_parquet(path)
    return path


def read_inseason_ratings(
    data_dir: Path,
    season: int,
    game_number: int | None = None,
) -> pl.DataFrame:
    """Read in-season ratings snapshot."""
    suffix = f"_game{game_number}" if game_number is not None else ""
    path = Path(data_dir) / "inseason_ratings" / f"inseason_ratings_{season}{suffix}.parquet"
    return pl.read_parquet(path)
