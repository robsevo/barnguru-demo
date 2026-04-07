# GRTZKY — NHL Hockey Simulation Engine

Possession-level NHL game simulator inspired by Haralabos Voulgaris's "Ewing" model.
See `CLAUDE.md` for full architecture and domain docs.

---

## Quick Start (existing machine)

### Rust engine
```bash
cargo test
```

### Python environment
```bash
uv sync --dev
uv run pytest tests/python/ -v
```

### Build PyO3 extension
```bash
uv run maturin develop --manifest-path engine/Cargo.toml
python -c "import gretzky_engine; print(gretzky_engine.gretzky_version())"
```

### Frontend (dev server)
```bash
cd dashboard/frontend && npm install && npm run dev
```

### Run a command
```bash
uv run python scripts/gretzky.py          # list all commands
uv run python scripts/gretzky.py sync     # sync live feeds
```

---

## Setting Up on a New Machine

### 1. Clone and install Python deps
```bash
git clone git@github.com:<username>/gretzky.git && cd gretzky
uv sync --dev
```

### 2. Build the Rust engine (required)
```bash
uv run maturin develop --manifest-path engine/Cargo.toml --release
python -c "import gretzky_engine; print(gretzky_engine.gretzky_version())"
```

### 3. Frontend deps
```bash
cd dashboard/frontend && npm install && cd ../..
```

### 4. Re-link Vercel (only needed for manual CLI deploys)
```bash
cd dashboard/frontend && vercel link && cd ../..
```
> If you only push to GitHub and let Vercel auto-deploy, skip this step — the GitHub integration handles it.

### 5. CV pipeline data (Phase 16 only — takes time and bandwidth)
```bash
uv run python scripts/gretzky.py build-cv-dataset   # downloads + extracts raw data (~5.7G)
uv run python scripts/gretzky.py train-cv-detector  # re-trains YOLO player detector
uv run python scripts/gretzky.py train-jersey-ocr   # re-trains jersey number OCR
```

---

## What's in the Repo vs What's Not

### In the repo (source of truth)
- All Python source: `data/`, `models/`, `models/cv/`, `scripts/`, `tests/`
- Rust engine source: `engine/src/`, `engine/python/`
- Frontend source: `dashboard/frontend/app/`, `utils/`, `public/`, config files
- Lookup tables and small JSON artifacts: `models/cv/team_color_lut.json`
- Config: `CLAUDE.md`, `PLAN.md`, `pyproject.toml`, `Cargo.toml`

### NOT in the repo (must restore)

| What | Why excluded | How to restore |
|------|-------------|----------------|
| `dashboard/frontend/node_modules/` | npm deps | `npm install` |
| `target/` | Rust build artifacts | `maturin develop` or `cargo build` |
| `data/cv_training/raw/` | 5.7G downloaded videos | `build-cv-dataset` |
| `data/cv_training/aug_images/` | 8.3G generated | `build-cv-dataset` |
| `data/cv_training/frames/` | 4.3G generated | `build-cv-dataset` |
| `data/cv_training/` (other dirs) | Generated pipeline output | `build-cv-dataset` |
| `runs/` | YOLO training logs + weights | `train-cv-detector` |
| `models/cv/*.pt` | Trained model weights | `train-cv-detector`, `train-jersey-ocr` |
| `yolov8n.pt` | Base YOLO weights | Auto-downloaded by ultralytics |
| `.env.local`, `.env.*.local` | Secrets | Set manually |
| `dashboard/frontend/.vercel/` | Local Vercel project link | `vercel link` |

---

## Vercel Deployment

Vercel auto-deploys on every push to `main` via GitHub integration (configured in Vercel dashboard — not in the repo).

- Production env vars (`NEXT_PUBLIC_API_URL`, etc.) live in **Vercel dashboard → Project → Settings → Environment Variables**
- FastAPI backend (`dashboard/api/`) runs locally or on a VPS — PyO3 binaries cannot run on Vercel serverless
- Build command: `npm run build` (auto-detected by Vercel)

---

## Project Structure

```
engine/          Rust: Markov state machine + Monte Carlo runner (PyO3 bindings)
models/          Python: xG, RAPM, fatigue, goalie, archetypes, CV models
data/            Python: NHL API, MoneyPuck, Polymarket, CV pipeline
dashboard/       Next.js frontend + FastAPI backend
backtest/        Season-level validation, Brier score, ROI curves
scripts/         CLI entry points (via gretzky.py dispatcher)
tests/           Rust unit tests + Python pytest
```
