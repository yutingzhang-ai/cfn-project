# cfn-project

A small PyTorch research codebase for **learning the flux function `F(u)` of a 1D
conservation law** from a precomputed simulation trajectory, using a **Conservative
Flux Network (CFN)** — a finite-volume neural surrogate that predicts numerical
fluxes on cell interfaces and integrates the conservation law in time.

The motivating physics is **Whitham modulation theory**: the analytical reference
flux is built from complete elliptic integrals `K(m), E(m)` and the goal is to
recover that closed-form `F(u)` directly from data, with no algebraic prior.

## What it does

For a 1D conservation law

$$ u_t + F(u)_x = 0 $$

the CFN parameterises the numerical flux at cell interfaces with a residual
1D-conv network and integrates the PDE with **forward Euler** or **TVD-RK3**.
Training is purely data-driven: given a long trajectory, random space-time
windows are sampled, the model is rolled out for `k` time steps, and the loss
is the MSE between the model's rolled-out states and the corresponding data
snapshots.

After training, `scripts/evaluate.py` compares the **learned** `F(u)` and
`dF/du` against the analytical Whitham flux, and rolls the trained CFN forward
side-by-side with a NumPy reference solver that uses the analytical flux on
the same grid.

## Install

```bash
pip install -e .[dev]
```

Or, to install only runtime deps:

```bash
pip install -e .
```

Python 3.10+ recommended.

## Quickstart

The repo ships with one bundled trajectory at `data/one_traj_loss_check.npy`
(shape `(1, Nt, Nx, 1)`) that the default config points at.

**Train:**

```bash
python scripts/train.py --config configs/default.yaml
```

This writes the following to the run dir (`runs/run_default/` by default):

| File | Contents |
|---|---|
| `model_init.pt` | Untrained model state (saved before training starts). |
| `model_trained.pt` | Best-loss model state. |
| `loss_history.npy` | Per-epoch training loss as a 1-D array. |
| `config.yaml` | The fully-resolved training config (CLI + YAML merged). |
| `train_summary.json` | `epochs_run`, `best_loss`, `final_lr`, `use_smooth_enabled`, `early_stopped`, `integrator`. |

**Evaluate:**

```bash
python scripts/evaluate.py --run-dir runs/run_default
```

`evaluate.py` reads `config.yaml` from the run dir, so you don't need to
re-pass `--features`/`--dt`/`--dx`. CLI flags still override.

It produces:

| File | What it shows |
|---|---|
| `loss_history.png` | Training loss curve (log y). |
| `flux_comparison.png` | `F(u)` and `dF/du` for the **untrained** model, the **trained** model, and the **analytical Whitham flux** — vertically aligned at a fixed anchor `u`. |
| `flux_multires.png` | The trained `F(u)` and `dF/du` resampled at `N ∈ {30, 100, 300}` cells. The CFN should be resolution-independent, so all rows should agree on shape. Derivatives use **autograd** (the model's true `dF/du`), finite differences, and the analytical reference. |
| `rollout_comparison.png` | Several `L`-step rollouts comparing the CFN (`TVD_RK3`) against a NumPy reference solver using the analytical flux, both initialised from the data IC. The grey strip shows the `margin` cells excluded from the RMSE metric. |
| `metrics.csv` | Flux/derivative RMSE for the single + multires sections, plus rollout RMSE per snapshot. |

`evaluate.py` also prints `Flux RMSE` and `Derivative RMSE` to stdout, plus
per-snapshot rollout RMSE (NumPy ref vs. CFN, both compared to data).

## Configuration

Configs live in `configs/`. Two are provided:

- `default.yaml` — short run (66 epochs, large space-time window).
- `long.yaml` — long fine-grained run (666 epochs, smaller windows, more rollout steps).

Any field can be overridden from the CLI; see `python scripts/train.py --help`.

Selected fields:

| Field | Meaning |
|---|---|
| `features` | Channel widths of the residual flux network. |
| `dt`, `dx` | Time/space step used inside the CFN. |
| `L` | Time-step gap between supervised snapshots. |
| `rollout_steps` | Number of supervised future points per training window. |
| `margin` | Cells excluded on each side of the loss (boundary effects). |
| `use_smooth`, `lambda_smooth` | Optional `(d/dx)(dF/du)` smoothness penalty. Auto-enabled the first time `ReduceLROnPlateau` drops the LR. |
| `integrator` | `euler` (default) or `rk3` (TVD-RK3). |

## Layout

```
cfn-project/
├── cfn/                        # core package
│   ├── models.py               # CFN, Flux, ResidualBlock
│   ├── padding.py              # circular + non-periodic ghost-cell padding
│   ├── data.py                 # random space-time window sampler
│   ├── training.py             # apply_model / train_epoch / train_CFN
│   ├── theoretical.py          # analytical Whitham flux
│   └── solvers.py              # NumPy reference TVD-RK3 / Euler
├── scripts/
│   ├── train.py                # YAML+CLI training entrypoint
│   └── evaluate.py             # post-training plots and metrics
├── configs/                    # default.yaml, long.yaml
├── data/                       # bundled trajectory + README
├── tests/                      # pytest suite (models, padding, solvers, theoretical, training)
└── .github/workflows/ci.yml    # ruff + pytest on push / PR
```

## Tests and lint

```bash
pytest -q
ruff check .
```

CI runs both on every push and pull request (`.github/workflows/ci.yml`).

## Notes / caveats

- Trajectory format is fixed at `(1, Nt, Nx, 1)` `float32`/`float64`.
- The boundary handling in both the CFN and the reference solver is
  **linear extrapolation**, so the two are directly comparable. The first/last
  few cells will always be inaccurate; the `margin` config controls how many
  are excluded from the loss/metrics.
- `evaluate.py` aligns the analytical flux to the learned flux at a single
  anchor `u` (midpoint of the 100-sample grid). The flux is only physically
  defined up to an additive constant, so the alignment is necessary; using a
  single global anchor makes the multires panels directly comparable.
- The CLI `merge_config` overrides any YAML field whose CLI flag has a default
  (effectively all of them). To pin a field via YAML, leave the CLI flag off.

## License

See `LICENSE`.
