"""Generate a viscous Burgers trajectory on a periodic domain.

Produces a .npy file of shape (1, Nt, Nx, 1) compatible with the existing
CFN training pipeline. The data is *viscous* Burgers (small viscosity
keeps it smooth), but the target flux for CFN remains F(u) = 1/2 u^2.

Usage
-----
    python scripts/generate_burgers_data.py \\
        --out data/burgers_traj.npy \\
        --Nx 512 --T 2.0 --nu 1e-3 --ic sin2x --dt-saved 0.005

Notes
-----
- Domain is [0, 2pi] with periodic BCs.
- Initial conditions:
    sin         : u0(x) = sin(x)
    sin2x       : u0(x) = sin(2x)     <- default
    smooth_step : 0.5 * (1 - tanh(5*(x - pi)))
- Spatial scheme: central flux for advection + central second difference
  for diffusion. With nu > 0 this is stable on smooth solutions.
- Time scheme: forward Euler with sub-stepping. The user specifies the
  desired *saved* time spacing via --dt-saved; the script picks an integer
  number of inner sub-steps so the inner step satisfies CFL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="Output .npy path.")
    p.add_argument("--Nx", type=int, default=512, help="Number of cells.")
    p.add_argument("--T", type=float, default=2.0, help="Final time.")
    p.add_argument("--nu", type=float, default=1e-3,
                   help="Viscosity. 1e-3 keeps the wave smooth past breaking (~t=1).")
    p.add_argument("--ic", choices=["sin", "sin2x", "smooth_step"], default="sin2x",
                   help="Initial condition. Default: sin2x.")
    p.add_argument("--dt-saved", type=float, default=0.005,
                   help="Time spacing between SAVED snapshots. Default: 0.005.")
    p.add_argument("--cfl", type=float, default=0.4,
                   help="Advective CFL target for inner sub-step.")
    return p.parse_args()


def rhs(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Periodic central-flux advection + central diffusion."""
    uL = np.roll(u, +1)
    uR = np.roll(u, -1)

    # Central numerical flux F(u) = 0.5 u^2 averaged at interfaces
    f = 0.5 * u * u
    f_R = np.roll(f, -1)
    f_L = np.roll(f, +1)
    flux_iphalf = 0.5 * (f + f_R)       # interface i+1/2
    flux_imhalf = 0.5 * (f_L + f)       # interface i-1/2
    advection = -(flux_iphalf - flux_imhalf) / dx

    # Diffusion: nu * u_xx
    diffusion = nu * (uR - 2.0 * u + uL) / (dx * dx)

    return advection + diffusion


def initial_condition(name: str, x: np.ndarray) -> np.ndarray:
    """Build u0(x) for the requested IC name."""
    if name == "sin":
        return np.sin(x).copy()
    if name == "sin2x":
        return np.sin(2.0 * x).copy()
    if name == "smooth_step":
        return 0.5 * (1.0 - np.tanh(5.0 * (x - np.pi)))
    raise ValueError(f"Unknown IC: {name!r}")


def main() -> None:
    args = parse_args()

    L = 2.0 * np.pi
    dx = L / args.Nx
    x = np.linspace(0.0, L, args.Nx, endpoint=False)

    u = initial_condition(args.ic, x)

    # Pick inner sub-step that respects CFL and divides dt_saved exactly.
    max_u = max(1.0, float(np.max(np.abs(u))))
    dt_adv = args.cfl * dx / max_u
    dt_diff = 0.5 * dx * dx / max(args.nu, 1e-12)
    dt_cfl = min(dt_adv, dt_diff)

    sub_steps = max(1, int(np.ceil(args.dt_saved / dt_cfl)))
    dt_inner = args.dt_saved / sub_steps

    Nt_saved = int(np.round(args.T / args.dt_saved)) + 1     # includes t=0
    Nt_inner = (Nt_saved - 1) * sub_steps

    print(f"IC               : {args.ic}")
    print(f"Domain           : [0, 2pi],  Nx={args.Nx},  dx={dx:.6e}")
    print(f"Viscosity nu     : {args.nu:.2e}")
    print(f"CFL limit dt     : {dt_cfl:.6e}  (adv={dt_adv:.3e}, diff={dt_diff:.3e})")
    print(f"dt_saved         : {args.dt_saved:.6e}  (requested)")
    print(f"sub_steps        : {sub_steps}")
    print(f"dt_inner         : {dt_inner:.6e}  (= dt_saved / sub_steps)")
    print(f"Nt_saved         : {Nt_saved}")
    print(f"Nt_inner total   : {Nt_inner}")

    # Integrate.
    snapshots = np.empty((Nt_saved, args.Nx), dtype=np.float64)
    snapshots[0] = u

    for k in range(1, Nt_saved):
        for _ in range(sub_steps):
            u = u + dt_inner * rhs(u, dx, args.nu)
        snapshots[k] = u

    print(f"u range over trajectory : [{snapshots.min():.4f}, {snapshots.max():.4f}]")

    # Reshape to CFN's expected (1, Nt, Nx, 1)
    traj = snapshots[np.newaxis, :, :, np.newaxis].astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, traj)
    print(f"Wrote {args.out}")
    print(f"  shape    = {traj.shape}")
    print(f"  dt_saved = {args.dt_saved:.6e}   (use this in YAML)")
    print(f"  dx       = {dx:.6e}              (use this in YAML)")


if __name__ == "__main__":
    main()
