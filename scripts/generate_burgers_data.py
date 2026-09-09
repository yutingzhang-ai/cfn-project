"""Generate inviscid Burgers trajectories on a periodic domain.

Shock-capturing scheme: Engquist-Osher (EO) flux + TVD-RK3 (Shu-Osher).
For Burgers (f(u) = 1/2 u^2), the EO flux collapses to the closed form

    F_EO(uL, uR) = 1/2 ( max(uL, 0)^2 + min(uR, 0)^2 )

which is fully vectorized -- no per-cell Python loop.

Produces a .npy file of shape (N_ic, Nt, Nx, 1), compatible with the
existing CFN training pipeline. By default a single trajectory is written
(N_ic=1); pass --num-ic > 1 to generate a dataset of randomized ICs.

Usage
-----
    # Single trajectory (drop-in replacement)
    python scripts/generate_burgers_data.py \\
        --out data/burgers_traj.npy \\
        --Nx 512 --T 1.5 --ic sin --dt-saved 0.005

    # Dataset of 200 random ICs (alpha + beta*sin(x))
    python scripts/generate_burgers_data.py \\
        --out data/burgers_train.npy \\
        --Nx 512 --T 1.5 --ic random_sin --num-ic 200 --seed-start 0

Notes
-----
- Domain is [0, 2pi] with periodic BCs (matches your existing pipeline).
- Initial conditions:
    sin         : u0(x) = sin(x)
    sin2x       : u0(x) = sin(2x)
    random_sin  : u0(x) = alpha + beta * sin(x), alpha~U[-eps,eps],
                  beta~U[1-eps,1+eps]   (matches your notebook)
- Optional viscosity --nu (default 0). With nu=0 this is inviscid Burgers;
  EO provides the numerical dissipation needed to capture the shock.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="Output .npy path.")
    p.add_argument("--Nx", type=int, default=512, help="Number of cells.")
    p.add_argument("--T", type=float, default=1.5, help="Final time.")
    p.add_argument("--nu", type=float, default=0.0,
                   help="Optional physical viscosity. Default 0 (inviscid).")
    p.add_argument("--ic", choices=["sin", "sin2x", "random_sin"],
                   default="sin",
                   help="Initial condition family.")
    p.add_argument("--num-ic", type=int, default=1,
                   help="Number of trajectories (only meaningful for random_sin).")
    p.add_argument("--seed-start", type=int, default=0,
                   help="Starting seed for random ICs.")
    p.add_argument("--eps-s", type=float, default=0.25,
                   help="Perturbation amplitude for random_sin IC.")
    p.add_argument("--dt-saved", type=float, default=0.005,
                   help="Time spacing between SAVED snapshots.")
    p.add_argument("--cfl", type=float, default=0.4,
                   help="Advective CFL target for inner sub-step.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Engquist-Osher flux for Burgers (vectorized closed form)
# ----------------------------------------------------------------------

def burgers_rhs(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Right-hand side du/dt for inviscid (+ optional viscous) Burgers.

    Uses EO flux at interfaces:
        F_{i+1/2} = 1/2 ( max(u_i, 0)^2 + min(u_{i+1}, 0)^2 )

    Periodic BCs via np.roll.
    """
    # Interface i+1/2 uses (u_i, u_{i+1}); roll(-1) gives u_{i+1}.
    uR = np.roll(u, -1)
    flux_iphalf = 0.5 * (np.maximum(u, 0.0) ** 2 + np.minimum(uR, 0.0) ** 2)
    # Interface i-1/2 is the same flux shifted one cell to the left.
    flux_imhalf = np.roll(flux_iphalf, +1)

    dudt = -(flux_iphalf - flux_imhalf) / dx

    if nu > 0.0:
        uL = np.roll(u, +1)
        dudt = dudt + nu * (uR - 2.0 * u + uL) / (dx * dx)

    return dudt


# ----------------------------------------------------------------------
# TVD-RK3 (Shu-Osher) time step
# ----------------------------------------------------------------------

def step_tvd_rk3(u: np.ndarray, dx: float, dt: float, nu: float) -> np.ndarray:
    """One TVD-RK3 step."""
    k1 = burgers_rhs(u, dx, nu)
    u1 = u + dt * k1

    k2 = burgers_rhs(u1, dx, nu)
    u2 = 0.75 * u + 0.25 * (u1 + dt * k2)

    k3 = burgers_rhs(u2, dx, nu)
    u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)
    return u_new


# ----------------------------------------------------------------------
# Initial conditions
# ----------------------------------------------------------------------

def initial_condition(name: str, x: np.ndarray, seed: int,
                      eps_s: float) -> np.ndarray:
    """Build u0(x) for the requested IC name. `seed` only used by random_sin."""
    if name == "sin":
        return np.sin(x).copy()
    if name == "sin2x":
        return np.sin(2.0 * x).copy()
    if name == "random_sin":
        rng = np.random.default_rng(seed)
        alpha = rng.uniform(-eps_s, eps_s)
        beta = rng.uniform(1.0 - eps_s, 1.0 + eps_s)
        return alpha + beta * np.sin(x)
    raise ValueError(f"Unknown IC: {name!r}")


# ----------------------------------------------------------------------
# Integration loop for a single trajectory
# ----------------------------------------------------------------------

def integrate_one(u0: np.ndarray, dx: float, Nt_saved: int,
                  dt_saved: float, cfl: float, nu: float) -> np.ndarray:
    """Integrate a single IC and return snapshots of shape (Nt_saved, Nx)."""
    u = u0.copy()

    # Pick inner sub-step that respects CFL and divides dt_saved exactly.
    max_u = max(1.0, float(np.max(np.abs(u))))
    dt_adv = cfl * dx / max_u
    if nu > 0.0:
        dt_diff = 0.5 * dx * dx / nu
        dt_cfl = min(dt_adv, dt_diff)
    else:
        dt_cfl = dt_adv

    sub_steps = max(1, int(np.ceil(dt_saved / dt_cfl)))
    dt_inner = dt_saved / sub_steps

    snapshots = np.empty((Nt_saved, u.size), dtype=np.float64)
    snapshots[0] = u
    for k in range(1, Nt_saved):
        for _ in range(sub_steps):
            u = step_tvd_rk3(u, dx, dt_inner, nu)
        snapshots[k] = u

    return snapshots, sub_steps, dt_inner


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    L = 2.0 * np.pi
    dx = L / args.Nx
    x = np.linspace(0.0, L, args.Nx, endpoint=False)

    Nt_saved = int(np.round(args.T / args.dt_saved)) + 1     # includes t=0

    num_ic = args.num_ic if args.ic == "random_sin" else 1

    print("Scheme           : Engquist-Osher flux + TVD-RK3")
    print(f"IC               : {args.ic}  (num_ic={num_ic})")
    print(f"Domain           : [0, 2pi],  Nx={args.Nx},  dx={dx:.6e}")
    print(f"Viscosity nu     : {args.nu:.2e}  (0 = inviscid)")
    print(f"dt_saved         : {args.dt_saved:.6e}")
    print(f"Nt_saved         : {Nt_saved}")
    print(f"Final time T     : {args.T}")

    all_traj = np.empty((num_ic, Nt_saved, args.Nx), dtype=np.float64)

    for i in range(num_ic):
        seed = args.seed_start + i
        u0 = initial_condition(args.ic, x, seed=seed, eps_s=args.eps_s)
        snaps, sub_steps, dt_inner = integrate_one(
            u0, dx, Nt_saved, args.dt_saved, args.cfl, args.nu,
        )
        all_traj[i] = snaps
        if i == 0:
            print(f"sub_steps        : {sub_steps}")
            print(f"dt_inner         : {dt_inner:.6e}")
        if num_ic > 1 and (i + 1) % max(1, num_ic // 10) == 0:
            print(f"  ... {i + 1}/{num_ic} trajectories done")

    print(f"u range over data : [{all_traj.min():.4f}, {all_traj.max():.4f}]")

    # Reshape to CFN's expected (N_ic, Nt, Nx, 1)
    traj = all_traj[..., np.newaxis].astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, traj)
    print(f"Wrote {args.out}")
    print(f"  shape    = {traj.shape}")
    print(f"  dt_saved = {args.dt_saved:.6e}   (use this in YAML)")
    print(f"  dx       = {dx:.6e}              (use this in YAML)")


if __name__ == "__main__":
    main()
