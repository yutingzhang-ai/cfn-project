"""Generate Whitham trajectories on a non-periodic (extrapolate) domain.

PDE: u_t + F(u)_x = 0, where F(u) is the Whitham modulation flux built
from elliptic integrals (see :func:`cfn.theoretical.build_flux`). F is a
smooth, monotone scalar flux over the registered operating range
``u in [u_min, u_max] = [0.66, 0.88]`` so the wave speed |F'(u)| stays
positive and finite; outside that range the table is clamped.

Boundaries
----------
The registered Whitham config uses ``boundary_mode = "extrapolate"`` --
non-periodic ghost-cell padding (zero-order extrapolation) for both the
trained CFN and the reference solver. This generator matches that, and
keeps the initial condition strictly inside the registered range so the
flux/wave-speed lookups never hit the clamped endpoints.

Scheme: Rusanov interface flux + TVD-RK3 (Shu-Osher), delegated to
``cfn.solvers.tvd_rk3_numpy(..., boundary_mode="extrapolate",
scheme="rusanov")`` so the generator and the trained-model reference
solver agree by construction.

Produces a .npy file of shape (1, Nt, Nx, 1) compatible with the CFN
training pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfn import theoretical
from cfn.solvers import tvd_rk3_numpy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="Output .npy path.")
    # Defaults below reproduce the historical Whitham trajectory shape that
    # ``configs/whitham.yaml`` is parameterized for: Nx=200, dx=0.4 (L=80),
    # dt_saved=100/3200=0.03125, T=100 -> Nt_saved=3201.
    p.add_argument("--Nx", type=int, default=200, help="Number of cells.")
    p.add_argument("--dx", type=float, default=0.4, help="Cell width.")
    p.add_argument("--T", type=float, default=100.0, help="Final time.")
    p.add_argument("--dt-saved", type=float, default=100.0 / 3200.0,
                   help="Time spacing between SAVED snapshots.")
    p.add_argument("--cfl", type=float, default=0.4,
                   help="CFL target for the inner sub-step.")
    p.add_argument("--ic", choices=["sin", "gauss"], default="sin",
                   help="Initial condition family.")
    p.add_argument("--mean", type=float, default=0.77,
                   help="IC mean. Default 0.77 is the centre of the "
                        "registered Whitham range [0.66, 0.88].")
    p.add_argument("--amp", type=float, default=0.08,
                   help="IC amplitude. Default 0.08 keeps u in [0.69, 0.85] "
                        "safely inside the registered range.")
    p.add_argument("--ic-wavelengths", type=float, default=1.0,
                   help="For --ic sin: number of full wavelengths across "
                        "the domain.")
    return p.parse_args()


def initial_condition(name: str, x: np.ndarray, mean: float, amp: float,
                      wavelengths: float) -> np.ndarray:
    """Return u0(x) of shape (Nx,) inside the registered Whitham range."""
    L = x[-1] - x[0] + (x[1] - x[0])  # treat x as cell-centres; period = L
    if name == "sin":
        return mean + amp * np.sin(2.0 * np.pi * wavelengths * x / L)
    if name == "gauss":
        c = L * 0.5
        sigma = L * 0.08
        return mean + amp * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    raise ValueError(f"Unknown IC: {name!r}")


def main() -> None:
    args = parse_args()

    # Activate the Whitham flux so rhs_flux_numpy / tvd_rk3_numpy use it.
    theoretical.set_equation("whitham")
    eq = theoretical.active_equation()

    L_domain = args.dx * args.Nx
    # Cell-centred grid on [0, L_domain] so the IC and the extrapolate
    # boundary ghosts see consistent values.
    x = (np.arange(args.Nx) + 0.5) * args.dx

    u0 = initial_condition(args.ic, x, args.mean, args.amp, args.ic_wavelengths)

    # Safety: clip-check that the IC is inside the registered range. We
    # don't auto-clip because silently shrinking the IC would surprise the
    # user; raising is the safer default.
    if float(u0.min()) < eq.u_min or float(u0.max()) > eq.u_max:
        raise SystemExit(
            f"Initial condition u in [{u0.min():.4f}, {u0.max():.4f}] is "
            f"outside the registered Whitham range [{eq.u_min}, {eq.u_max}]. "
            f"Reduce --amp or pick a --mean closer to the centre."
        )

    U = u0.copy()[None, ..., None]  # (1, Nx, 1) for solver convention

    # Choose dt so that dt * max|F'(u)| / dx <= cfl. The wave speed varies
    # with u, so we refresh the CFL once per saved snapshot rather than
    # locking it from the IC alone -- under extrapolate BCs the rollout can
    # drift outside the registered range over long T, in which case the
    # table-clamped wave speed estimate would otherwise be stale.
    def cfl_substep_count(u: np.ndarray) -> tuple[int, float]:
        speed = float(eq.wave_speed(u).max())
        if not np.isfinite(speed) or speed <= 0.0:
            raise SystemExit(
                f"Whitham wave speed non-positive or non-finite ({speed}); "
                f"unexpected -- check the operating range."
            )
        dt_cfl = args.cfl * args.dx / speed
        n = max(1, int(np.ceil(args.dt_saved / dt_cfl)))
        return n, args.dt_saved / n

    sub_steps0, dt_inner0 = cfl_substep_count(u0)
    speed_init = float(eq.wave_speed(u0).max())

    Nt_saved = int(np.round(args.T / args.dt_saved)) + 1
    snaps = np.empty((Nt_saved, args.Nx), dtype=np.float64)
    snaps[0] = u0

    print("Whitham generator (Rusanov via cfn.solvers + TVD-RK3, extrapolate BCs)")
    print(f"  Nx={args.Nx}  dx={args.dx:.4f}  L={L_domain:.3f}")
    print(f"  IC               : {args.ic}  mean={args.mean}  amp={args.amp}")
    print(f"  IC range         : [{u0.min():.4f}, {u0.max():.4f}]  "
          f"(registered: [{eq.u_min}, {eq.u_max}])")
    print(f"  init |F'(u)|_max = {speed_init:.4f}")
    print(f"  dt_saved={args.dt_saved:.6e}  initial sub_steps={sub_steps0}  "
          f"dt_inner={dt_inner0:.4e}")
    print(f"  Nt_saved         : {Nt_saved}   (T={args.T})")

    rolling_min = float(u0.min())
    rolling_max = float(u0.max())
    sub_steps_now = sub_steps0
    dt_inner_now = dt_inner0
    for k in range(1, Nt_saved):
        # Refresh CFL once per saved snapshot based on the current state.
        sub_steps_now, dt_inner_now = cfl_substep_count(U[0, :, 0])
        for _ in range(sub_steps_now):
            U = tvd_rk3_numpy(U, dt_inner_now, args.dx,
                              boundary_mode="extrapolate", scheme="rusanov")
        snaps[k] = U[0, :, 0]
        rolling_min = min(rolling_min, float(snaps[k].min()))
        rolling_max = max(rolling_max, float(snaps[k].max()))

    final = snaps[-1]
    print(f"  final u in [{final.min():.4f}, {final.max():.4f}]   "
          f"(extrapolate BCs are not conservative; small drift expected)")
    print(f"  rollout u range  : [{rolling_min:.4f}, {rolling_max:.4f}]")
    if rolling_min < eq.u_min or rolling_max > eq.u_max:
        print(f"  WARNING: u left the registered Whitham range "
              f"[{eq.u_min}, {eq.u_max}] during the rollout. Outside that "
              f"range the elliptic-integral flux/wave-speed tables are "
              f"clamped, so the trajectory is no longer a faithful Whitham "
              f"solution. Consider reducing --T or --amp, switching to "
              f"--ic gauss with a narrower bump, or extending the operating "
              f"range in cfn.theoretical.")

    # CFN convention: (1, Nt, Nx, 1)
    traj = snaps[None, ..., None].astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, traj)
    print(f"Wrote {args.out}  shape={traj.shape}  dx={args.dx:.6e}  "
          f"dt_saved={args.dt_saved}")


if __name__ == "__main__":
    main()
