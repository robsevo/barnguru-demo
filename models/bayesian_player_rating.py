"""Bayesian Player Rating System — Features 2.9 + 2.10.

Feature 2.9 — Hierarchical Bayesian model for per-player posterior
distributions with credible intervals.  Handles small samples (AHL
call-ups, rookies) via shrinkage toward the league-mean prior.

Feature 2.10 — Real-time sequential Bayesian update using the
Normal-Normal conjugate model.  After each game result, the posterior
moves with the evidence but stays anchored by the prior.

The two features share one module:
    - ``BayesianPlayerRating`` — batch / season-level posterior
    - ``SequentialBayesianUpdater`` — per-game streaming update

No PyMC required for the conjugate path.  PyMC is used only when
``use_pymc=True`` and the library is available — useful for full
hierarchical inference on large samples.

Normal-Normal conjugate update (Feature 2.10):
    Prior:  θ ~ N(μ₀, σ₀²)
    Likelihood: y | θ ~ N(θ, σ_obs²)
    Posterior precision: 1/σ₁² = 1/σ₀² + n/σ_obs²
    Posterior mean:      μ₁ = σ₁² × (μ₀/σ₀² + Σy/σ_obs²)

Outputs
-------
``player_ratings_{season}.parquet`` — one row per player:
    posterior_mean    posterior mean xGF/60 rating
    posterior_sigma   posterior standard deviation
    ci_lower_95       2.5th percentile
    ci_upper_95       97.5th percentile
    credible_width    ci_upper_95 − ci_lower_95 (uncertainty)
    n_games           number of games observed
    model_version     "bayes_v1"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning

try:
    import pymc as pm  # type: ignore
    _PYMC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYMC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "bayes_v1"

# League-level hyper-priors (xGF/60 at EV)
LEAGUE_MEAN_XGF60    = 2.50   # league average xGF/60 for forwards
LEAGUE_SIGMA_XGF60   = 0.60   # cross-player standard deviation
OBS_SIGMA_XGF60      = 1.20   # per-game noise (high variance in individual games)
ROOKIE_SIGMA_INFLATE = 1.5    # inflate sigma for call-ups / < 10 GP


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

PLAYER_RATING_SCHEMA: dict[str, pl.DataType] = {
    "player_id":      pl.Int64,
    "player_name":    pl.Utf8,
    "season":         pl.Int64,
    "n_games":        pl.Int64,
    "prior_mean":     pl.Float64,
    "prior_sigma":    pl.Float64,
    "posterior_mean": pl.Float64,
    "posterior_sigma": pl.Float64,
    "ci_lower_95":    pl.Float64,
    "ci_upper_95":    pl.Float64,
    "credible_width": pl.Float64,
    "model_version":  pl.Utf8,
}


# ---------------------------------------------------------------------------
# Conjugate update (Feature 2.10 core)
# ---------------------------------------------------------------------------

def conjugate_update(
    prior_mu: float,
    prior_sigma: float,
    observations: np.ndarray,
    obs_sigma: float = OBS_SIGMA_XGF60,
) -> tuple[float, float]:
    """Normal-Normal conjugate Bayesian update.

    Args:
        prior_mu:     Prior mean.
        prior_sigma:  Prior standard deviation (> 0).
        observations: 1-D array of observed values (e.g. per-game xGF/60).
        obs_sigma:    Observation noise std (assumed known).

    Returns:
        (posterior_mu, posterior_sigma) — updated parameters.
    """
    if prior_sigma <= 0:
        raise ValueError(f"prior_sigma must be > 0, got {prior_sigma}")
    if obs_sigma <= 0:
        raise ValueError(f"obs_sigma must be > 0, got {obs_sigma}")

    n = len(observations)
    if n == 0:
        return prior_mu, prior_sigma

    prior_prec   = 1.0 / (prior_sigma ** 2)
    obs_prec     = n / (obs_sigma ** 2)
    post_prec    = prior_prec + obs_prec
    post_sigma   = float(np.sqrt(1.0 / post_prec))
    post_mu      = float(
        (prior_mu * prior_prec + observations.sum() * (1.0 / obs_sigma ** 2))
        / post_prec
    )
    return post_mu, post_sigma


# ---------------------------------------------------------------------------
# SequentialBayesianUpdater — Feature 2.10
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Per-player Bayesian state (mutable, updated after each game).

    Attributes:
        player_id:   NHL player ID.
        player_name: Display name.
        mu:          Current posterior mean.
        sigma:       Current posterior standard deviation.
        n_games:     Number of game observations processed.
    """
    player_id:   int
    player_name: str
    mu:          float
    sigma:       float
    n_games:     int = 0


class SequentialBayesianUpdater:
    """Real-time per-game Bayesian rating update (Feature 2.10).

    Maintains a running posterior per player.  Call ``update()`` after
    each game to incorporate the observed xGF/60 value.

    Usage::

        updater = SequentialBayesianUpdater()
        # After game 1:
        updater.update(player_id=8478402, player_name="Matthews", observation=3.1)
        # After game 2:
        updater.update(player_id=8478402, player_name="Matthews", observation=2.7)
        state = updater.get(8478402)
        print(f"μ={state.mu:.3f}  σ={state.sigma:.3f}")
    """

    def __init__(
        self,
        league_mean:  float = LEAGUE_MEAN_XGF60,
        league_sigma: float = LEAGUE_SIGMA_XGF60,
        obs_sigma:    float = OBS_SIGMA_XGF60,
    ) -> None:
        self._league_mean  = league_mean
        self._league_sigma = league_sigma
        self._obs_sigma    = obs_sigma
        self._states: dict[int, PlayerState] = {}

    # ------------------------------------------------------------------

    def update(
        self,
        player_id:   int,
        player_name: str,
        observation: float,
        obs_sigma:   float | None = None,
    ) -> PlayerState:
        """Incorporate one game observation and return updated state.

        Args:
            player_id:   Player identifier.
            player_name: Display name (used for initialisation).
            observation: Observed metric for this game (e.g. xGF/60).
            obs_sigma:   Override per-game noise (default: self._obs_sigma).

        Returns:
            Updated PlayerState.
        """
        if obs_sigma is None:
            obs_sigma = self._obs_sigma

        state = self._states.get(player_id)
        if state is None:
            state = PlayerState(
                player_id   = player_id,
                player_name = player_name,
                mu          = self._league_mean,
                sigma       = self._league_sigma,
                n_games     = 0,
            )

        post_mu, post_sigma = conjugate_update(
            prior_mu     = state.mu,
            prior_sigma  = state.sigma,
            observations = np.array([observation]),
            obs_sigma    = obs_sigma,
        )
        state.mu      = post_mu
        state.sigma   = post_sigma
        state.n_games += 1
        state.player_name = player_name
        self._states[player_id] = state
        return state

    def update_batch(
        self,
        player_id:    int,
        player_name:  str,
        observations: np.ndarray,
        obs_sigma:    float | None = None,
    ) -> PlayerState:
        """Incorporate multiple observations at once (efficient conjugate update).

        Equivalent to calling update() in a loop but processes all
        observations in a single precision-weighted computation.
        """
        if obs_sigma is None:
            obs_sigma = self._obs_sigma

        state = self._states.get(player_id)
        if state is None:
            state = PlayerState(
                player_id   = player_id,
                player_name = player_name,
                mu          = self._league_mean,
                sigma       = self._league_sigma,
                n_games     = 0,
            )

        post_mu, post_sigma = conjugate_update(
            prior_mu     = state.mu,
            prior_sigma  = state.sigma,
            observations = observations,
            obs_sigma    = obs_sigma,
        )
        state.mu       = post_mu
        state.sigma    = post_sigma
        state.n_games += len(observations)
        state.player_name = player_name
        self._states[player_id] = state
        return state

    def get(self, player_id: int) -> PlayerState | None:
        """Return current state for a player, or None if unknown."""
        return self._states.get(player_id)

    def all_states(self) -> list[PlayerState]:
        """Return all player states, sorted by posterior mean descending."""
        return sorted(self._states.values(), key=lambda s: s.mu, reverse=True)

    def reset(self, player_id: int | None = None) -> None:
        """Reset one player (or all) to league-mean prior."""
        if player_id is not None:
            self._states.pop(player_id, None)
        else:
            self._states.clear()

    def to_dataframe(self, season: int) -> pl.DataFrame:
        """Export all states to a PLAYER_RATING_SCHEMA-compatible DataFrame."""
        states = self.all_states()
        if not states:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYER_RATING_SCHEMA.items()}
            )
        z95 = 1.959964
        rows = {
            "player_id":       [s.player_id      for s in states],
            "player_name":     [s.player_name     for s in states],
            "season":          [season            for s in states],
            "n_games":         [s.n_games         for s in states],
            "prior_mean":      [self._league_mean for s in states],
            "prior_sigma":     [self._league_sigma for s in states],
            "posterior_mean":  [s.mu              for s in states],
            "posterior_sigma": [s.sigma           for s in states],
            "ci_lower_95":     [s.mu - z95 * s.sigma for s in states],
            "ci_upper_95":     [s.mu + z95 * s.sigma for s in states],
            "credible_width":  [2 * z95 * s.sigma    for s in states],
            "model_version":   [MODEL_VERSION     for s in states],
        }
        return pl.DataFrame(rows).cast(PLAYER_RATING_SCHEMA)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "states":       self._states,
            "league_mean":  self._league_mean,
            "league_sigma": self._league_sigma,
            "obs_sigma":    self._obs_sigma,
            "version":      MODEL_VERSION,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "SequentialBayesianUpdater":
        payload = joblib.load(Path(path))
        obj = cls(
            league_mean  = payload.get("league_mean",  LEAGUE_MEAN_XGF60),
            league_sigma = payload.get("league_sigma", LEAGUE_SIGMA_XGF60),
            obs_sigma    = payload.get("obs_sigma",    OBS_SIGMA_XGF60),
        )
        obj._states = payload.get("states", {})
        return obj


# ---------------------------------------------------------------------------
# BayesianPlayerRating — Feature 2.9 (batch / season-level)
# ---------------------------------------------------------------------------

class BayesianPlayerRating:
    """Hierarchical Bayesian player rating model (Feature 2.9).

    Fits posterior distributions per player using:
    - Conjugate Normal-Normal update (default, fast, no deps beyond numpy)
    - Optional PyMC hierarchical model (``use_pymc=True``) for full
      posterior with learned league-mean and sigma hyper-parameters.

    Usage::

        model = BayesianPlayerRating()
        ratings = model.fit_season(
            player_game_df=game_log_df,  # player_id, xgf_per60, n_games
            season=2025,
        )
        model.save(Path("~/.gretzky/models/bayes_ratings_2025.pkl"))
    """

    def __init__(
        self,
        league_mean:  float = LEAGUE_MEAN_XGF60,
        league_sigma: float = LEAGUE_SIGMA_XGF60,
        obs_sigma:    float = OBS_SIGMA_XGF60,
        use_pymc:     bool  = False,
        pymc_samples: int   = 1000,
        pymc_chains:  int   = 2,
    ) -> None:
        self._league_mean  = league_mean
        self._league_sigma = league_sigma
        self._obs_sigma    = obs_sigma
        self._use_pymc     = use_pymc and _PYMC_AVAILABLE
        self._pymc_samples = pymc_samples
        self._pymc_chains  = pymc_chains
        self._version      = MODEL_VERSION
        self._ratings: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_season(
        self,
        player_game_df: pl.DataFrame,
        season: int,
        rapm_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute per-player posteriors for one season.

        Args:
            player_game_df: One row per player with columns:
                player_id, player_name (optional), xgf_per60 (mean observed),
                n_games (number of games), xgf_std (optional per-player std).
            season:         NHL season start-year.
            rapm_df:        Optional RAPM output to sharpen prior mean per
                            player (uses off_rapm as additive prior shift).

        Returns:
            DataFrame matching PLAYER_RATING_SCHEMA, sorted by posterior_mean desc.
        """
        required = {"player_id", "xgf_per60", "n_games"}
        missing = required - set(player_game_df.columns)
        if missing:
            raise ValueError(f"player_game_df missing columns: {sorted(missing)}")

        df = player_game_df.filter(pl.col("n_games") > 0)

        if df.is_empty():
            warnings.warn(
                f"Season {season}: no players with n_games > 0.",
                DataMissingWarning,
                stacklevel=2,
            )
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYER_RATING_SCHEMA.items()}
            )

        # Build RAPM prior offset map
        rapm_offset: dict[int, float] = {}
        if rapm_df is not None and "player_id" in rapm_df.columns:
            rapm_col = "off_rapm" if "off_rapm" in rapm_df.columns else None
            if rapm_col:
                for row in rapm_df.select(["player_id", rapm_col]).to_dicts():
                    pid = int(row["player_id"])
                    val = row.get(rapm_col)
                    if val is not None and not np.isnan(float(val)):
                        rapm_offset[pid] = float(val) * 0.30  # blend 30% RAPM signal

        if self._use_pymc:
            return self._fit_pymc(df, season, rapm_offset)
        return self._fit_conjugate(df, season, rapm_offset)

    def _fit_conjugate(
        self,
        df: pl.DataFrame,
        season: int,
        rapm_offset: dict[int, float],
    ) -> pl.DataFrame:
        """Conjugate Normal-Normal batch update for all players."""
        z95 = 1.959964
        player_ids   = df["player_id"].cast(pl.Int64).to_list()
        xgf_per60    = df["xgf_per60"].cast(pl.Float64).to_numpy()
        n_games_arr  = df["n_games"].cast(pl.Int64).to_numpy()
        player_names = (
            df["player_name"].to_list()
            if "player_name" in df.columns
            else [""] * len(df)
        )

        post_means  = []
        post_sigmas = []
        prior_means = []
        prior_sigmas = []

        for i, pid in enumerate(player_ids):
            prior_mu = self._league_mean + rapm_offset.get(pid, 0.0)
            # Inflate sigma for small-sample players (< 10 GP)
            n = int(n_games_arr[i])
            prior_sigma = (
                self._league_sigma * ROOKIE_SIGMA_INFLATE
                if n < 10
                else self._league_sigma
            )

            obs = np.full(n, xgf_per60[i])  # use per-game mean for each game slot
            post_mu, post_sigma = conjugate_update(
                prior_mu    = prior_mu,
                prior_sigma = prior_sigma,
                observations = obs,
                obs_sigma   = self._obs_sigma,
            )
            prior_means.append(prior_mu)
            prior_sigmas.append(prior_sigma)
            post_means.append(post_mu)
            post_sigmas.append(post_sigma)

        post_means_arr  = np.array(post_means)
        post_sigmas_arr = np.array(post_sigmas)

        result = pl.DataFrame({
            "player_id":       pl.Series(player_ids,               dtype=pl.Int64),
            "player_name":     pl.Series(player_names,             dtype=pl.Utf8),
            "season":          pl.Series([season] * len(df),       dtype=pl.Int64),
            "n_games":         pl.Series(n_games_arr.tolist(),     dtype=pl.Int64),
            "prior_mean":      pl.Series(prior_means,              dtype=pl.Float64),
            "prior_sigma":     pl.Series(prior_sigmas,             dtype=pl.Float64),
            "posterior_mean":  pl.Series(post_means_arr.tolist(),  dtype=pl.Float64),
            "posterior_sigma": pl.Series(post_sigmas_arr.tolist(), dtype=pl.Float64),
            "ci_lower_95":     pl.Series((post_means_arr - z95 * post_sigmas_arr).tolist(), dtype=pl.Float64),
            "ci_upper_95":     pl.Series((post_means_arr + z95 * post_sigmas_arr).tolist(), dtype=pl.Float64),
            "credible_width":  pl.Series((2 * z95 * post_sigmas_arr).tolist(), dtype=pl.Float64),
            "model_version":   pl.Series([MODEL_VERSION] * len(df), dtype=pl.Utf8),
        })
        self._ratings = result.sort("posterior_mean", descending=True)
        return self._ratings

    def _fit_pymc(
        self,
        df: pl.DataFrame,
        season: int,
        rapm_offset: dict[int, float],
    ) -> pl.DataFrame:  # pragma: no cover
        """Full hierarchical PyMC model (optional — requires pymc)."""
        if not _PYMC_AVAILABLE:
            warnings.warn(
                "PyMC not available — falling back to conjugate update.",
                ImportWarning,
                stacklevel=3,
            )
            return self._fit_conjugate(df, season, rapm_offset)

        import pymc as pm  # noqa: F401

        xgf_per60   = df["xgf_per60"].cast(pl.Float64).to_numpy()
        n_games_arr = df["n_games"].cast(pl.Int64).to_numpy().astype(int)

        # Build index array: repeat each player's xGF n_games times
        obs_list: list[float] = []
        player_idx: list[int] = []
        for i, (xgf, n) in enumerate(zip(xgf_per60, n_games_arr)):
            obs_list.extend([xgf] * n)
            player_idx.extend([i] * n)

        obs_arr = np.array(obs_list)
        pidx_arr = np.array(player_idx, dtype=int)

        with pm.Model():
            # Hyper-priors
            mu_mu    = pm.Normal("mu_mu",    mu=self._league_mean,  sigma=0.5)
            sigma_mu = pm.HalfNormal("sigma_mu", sigma=self._league_sigma)

            # Per-player quality
            theta = pm.Normal("theta", mu=mu_mu, sigma=sigma_mu, shape=len(df))

            # Likelihood
            pm.Normal("obs", mu=theta[pidx_arr], sigma=self._obs_sigma, observed=obs_arr)

            idata = pm.sample(
                draws=self._pymc_samples,
                chains=self._pymc_chains,
                progressbar=False,
            )

        theta_post = idata.posterior["theta"].values.reshape(-1, len(df))
        post_means  = theta_post.mean(axis=0)
        post_sigmas = theta_post.std(axis=0)

        z95 = 1.959964
        player_ids   = df["player_id"].cast(pl.Int64).to_list()
        player_names = (
            df["player_name"].to_list() if "player_name" in df.columns else [""] * len(df)
        )

        result = pl.DataFrame({
            "player_id":       pl.Series(player_ids,                               dtype=pl.Int64),
            "player_name":     pl.Series(player_names,                             dtype=pl.Utf8),
            "season":          pl.Series([season] * len(df),                       dtype=pl.Int64),
            "n_games":         pl.Series(n_games_arr.tolist(),                     dtype=pl.Int64),
            "prior_mean":      pl.Series([self._league_mean] * len(df),            dtype=pl.Float64),
            "prior_sigma":     pl.Series([self._league_sigma] * len(df),           dtype=pl.Float64),
            "posterior_mean":  pl.Series(post_means.tolist(),                      dtype=pl.Float64),
            "posterior_sigma": pl.Series(post_sigmas.tolist(),                     dtype=pl.Float64),
            "ci_lower_95":     pl.Series((post_means - z95 * post_sigmas).tolist(), dtype=pl.Float64),
            "ci_upper_95":     pl.Series((post_means + z95 * post_sigmas).tolist(), dtype=pl.Float64),
            "credible_width":  pl.Series((2 * z95 * post_sigmas).tolist(),          dtype=pl.Float64),
            "model_version":   pl.Series([MODEL_VERSION + "_pymc"] * len(df),      dtype=pl.Utf8),
        })
        self._ratings = result.sort("posterior_mean", descending=True)
        return self._ratings

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def ratings(self) -> pl.DataFrame:
        """Return last computed ratings DataFrame."""
        if self._ratings is None:
            raise RuntimeError("Call fit_season() first.")
        return self._ratings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "ratings":      self._ratings,
            "league_mean":  self._league_mean,
            "league_sigma": self._league_sigma,
            "obs_sigma":    self._obs_sigma,
            "version":      self._version,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "BayesianPlayerRating":
        payload = joblib.load(Path(path))
        obj = cls(
            league_mean  = payload.get("league_mean",  LEAGUE_MEAN_XGF60),
            league_sigma = payload.get("league_sigma", LEAGUE_SIGMA_XGF60),
            obs_sigma    = payload.get("obs_sigma",    OBS_SIGMA_XGF60),
        )
        obj._ratings = payload.get("ratings")
        obj._version = payload.get("version", MODEL_VERSION)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_player_ratings(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write Bayesian ratings to ``output_dir/player_ratings_{season}.parquet``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"player_ratings_{season}.parquet"
    for col, dtype in PLAYER_RATING_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(PLAYER_RATING_SCHEMA.keys())).write_parquet(path)
    return path


def read_player_ratings(data_dir: Path, season: int) -> pl.DataFrame:
    """Read Bayesian ratings from ``data_dir/bayes_ratings/player_ratings_{season}.parquet``."""
    path = Path(data_dir) / "bayes_ratings" / f"player_ratings_{season}.parquet"
    return pl.read_parquet(path)
