# cfn-project

A small PyTorch research codebase for **learning the flux function `F(u)` of a
1D conservation law `u_t + F(u)_x = 0`** from a precomputed simulation
trajectory, using a **Conservative Flux Network (CFN)** — a finite-volume
neural surrogate that predicts numerical fluxes on cell interfaces and
integrates the PDE in time. The motivating physics is **Whitham modulation
theory**, whose closed-form flux (built from elliptic integrals `K(m), E(m)`)
serves as the analytical reference.

## Install

```bash
pip install -e .[dev]      # runtime + tests + ruff
```

Python 3.10+ recommended.

## Quickstart

```bash
python scripts/train.py    --config configs/default.yaml
python scripts/evaluate.py --run-dir runs/run_default
```

`evaluate.py` reads `config.yaml` from the run dir, so model/physics
parameters do not need to be re-passed on the CLI.

## Run-dir outputs

| File | What it is |
|---|---|
| `model_init.pt` / `model_trained.pt` | Untrained and best-loss state dicts. |
| `loss_history.npy`, `loss_history.png` | Per-epoch training loss. |
| `config.yaml`, `train_summary.json` | Resolved config + final stats (epochs, best loss, integrator, etc.). |
| `flux_comparison.png` | `F(u)` and `dF/du` before / after / analytical, aligned at one anchor. |
| `flux_multires.png` | Same comparison resampled at `N ∈ {30, 100, 300}`; rows should agree. |
| `rollout_comparison.png` | CFN (`TVD_RK3`) vs. NumPy reference solver vs. data, several `L`-step snapshots. |
| `metrics.csv` | Flux / derivative / rollout RMSE in one place. |

## Layout

```
cfn/         core package: models, padding, data, training, theoretical, solvers
scripts/     train.py, evaluate.py
configs/     default.yaml (short), long.yaml (long fine run)
data/        bundled trajectory (1, Nt, Nx, 1)
tests/       pytest suite (models, padding, solvers, theoretical, training)
.github/     CI workflow (ruff + pytest)
```

## Configuration

Two configs are provided; any field can be overridden from the CLI
(`python scripts/train.py --help`). Notable fields: `features` (network
widths), `dt`/`dx`, `L` (snapshot gap), `rollout_steps`, `margin`,
`use_smooth`/`lambda_smooth` (auto-enabled on first LR drop), `integrator`
(`euler` or `rk3`).

## Tests

```bash
pytest -q
ruff check .
```

Both run in CI on every push / PR.

## License

See `LICENSE`.
