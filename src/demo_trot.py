"""Open-loop hand-tuned CPG trot. No learning, no policy - the weeks 1-2 milestone.

The oscillator setpoints (mu, omega) are held constant here. In week 3+ the PPO
policy replaces those constants and nothing else about this pipeline changes:

    CPG.step  ->  foot_targets  ->  legs_ik  ->  data.ctrl  ->  joint PD

Constants are integrated at the physics rate, matching CPG-RL, where the
oscillators run on the simulation clock and only the setpoints update at the
slower policy rate.

Usage:
    python src/demo_trot.py                    # watch it in the viewer
    python src/demo_trot.py --headless         # print numbers instead
    python src/demo_trot.py --gait bound --freq 3.0
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

import cpg
from go1_model import legs_ik, load_model

# Reported instead of episode reward, per CLAUDE.md. These are open-loop
# diagnostics for tuning, not results.
FALL_HEIGHT = 0.15  # trunk below this counts as fallen


def run(args) -> dict:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    osc = cpg.CPG(cpg.CPGConfig(gait=args.gait))
    cart = cpg.CartesianConfig(
        step_length=args.step_length,
        ground_clearance=args.clearance,
    )
    mu = np.full(cpg.N_LEGS, args.mu)
    omega = np.full(cpg.N_LEGS, 2.0 * np.pi * args.freq)

    dt = model.opt.timestep
    n_steps = int(args.seconds / dt)
    start_xy = data.qpos[:2].copy()
    heights, fell = [], False

    viewer = None
    if not args.headless:
        # Imported lazily and under an alias: `import mujoco.viewer` here would
        # rebind `mujoco` as a function local and shadow the module above.
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    for _ in range(n_steps):
        wall = time.time()
        osc.step(dt, mu, omega)
        data.ctrl[:] = legs_ik(osc.foot_targets(cart))
        mujoco.mj_step(model, data)

        heights.append(float(data.qpos[2]))
        if data.qpos[2] < FALL_HEIGHT:
            fell = True
            if args.stop_on_fall:
                break

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            lag = dt - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)

    if viewer is not None:
        viewer.close()

    displacement = data.qpos[:2] - start_xy
    return {
        "gait": args.gait,
        "seconds": data.time,
        "forward_velocity": float(displacement[0] / max(data.time, 1e-9)),
        "lateral_drift": float(displacement[1]),
        "mean_height": float(np.mean(heights)) if heights else float("nan"),
        "fell": fell,
        "phase_lags_over_pi": np.round(osc.phase_lags() / np.pi, 3),
        "amplitudes": np.round(osc.r, 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gait", default="trot", choices=sorted(cpg.GAITS))
    p.add_argument("--freq", type=float, default=2.0, help="stride frequency in Hz")
    p.add_argument("--mu", type=float, default=1.0, help="amplitude setpoint")
    p.add_argument("--step-length", type=float, default=0.06)
    p.add_argument("--clearance", type=float, default=0.05)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--stop-on-fall", action="store_true")
    args = p.parse_args()

    out = run(args)
    for k, v in out.items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()
