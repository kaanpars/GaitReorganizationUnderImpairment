"""Check the analytic leg IK against MuJoCo's own forward kinematics.

If IK(FK(q)) != q, every foot position the CPG commands is silently wrong, the
robot still walks, and the gait we report is not the gait the CPG asked for.
That is precisely the "silent env bug" failure mode in CLAUDE.md, so it gets a
test rather than an eyeball.

Run with:  python tests/test_ik.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cpg  # noqa: E402
from go1_model import (  # noqa: E402
    LEGS,
    L_CALF,
    L_THIGH,
    Y_SIGN,
    foot_ik,
    legs_ik,
    load_model,
)


def foot_in_hip_frame(model, data, leg: str) -> np.ndarray:
    """Foot position relative to the hip origin, expressed in the TRUNK frame.

    Not the hip body's own frame: that frame rotates with the abduction joint,
    so expressing the foot in it cancels the abduction angle out and no IK could
    recover it. The IK is defined in the fixed frame the abduction joint rotates
    within, which is the trunk's.
    """
    hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
    rel = data.site_xpos[foot] - data.xpos[hip]
    return data.xmat[trunk].reshape(3, 3).T @ rel


def leg_extension(q_thigh: float, q_calf: float) -> float:
    """In-plane vertical extension of the leg. Negative = foot below the hip."""
    return -(L_THIGH * np.cos(q_thigh) + L_CALF * np.cos(q_thigh + q_calf))


def test_ik_round_trip(n: int = 500, seed: int = 0) -> None:
    """IK inverts FK exactly on the branch it declares.

    A foot position does not determine the leg configuration on its own: the same
    point is reachable both with the leg extended downward and with it folded up
    over the hip, at a different abduction angle. `foot_ik` always returns the
    downward branch, so poses on the folded-up branch are excluded here. The next
    test shows the CPG never asks for one.
    """
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # trunk upright, so FK is readable
    rng = np.random.default_rng(seed)

    lo, hi = model.jnt_range[1:].T  # skip the free joint
    worst, tested, skipped = 0.0, 0, 0
    for _ in range(n):
        q = rng.uniform(lo, hi)
        data.qpos[7:] = q
        mujoco.mj_forward(model, data)
        for i, leg in enumerate(LEGS):
            q_leg = q[3 * i:3 * i + 3]
            if leg_extension(q_leg[1], q_leg[2]) >= 0.0:
                skipped += 1
                continue
            recovered = foot_ik(foot_in_hip_frame(model, data, leg), Y_SIGN[i])
            worst = max(worst, float(np.abs(recovered - q_leg).max()))
            tested += 1

    print(f"  round trip: {tested} poses on-branch ({skipped} folded-up poses skipped), "
          f"worst error {worst:.2e} rad")
    assert worst < 1e-9, f"IK does not invert FK (worst error {worst:.3e} rad)"


def test_cpg_targets_are_reachable(seconds: float = 3.0, dt: float = 0.002) -> None:
    """Over a full gait cycle, the foot lands where the CPG asked, within limits.

    This is the end-to-end check: CPG -> Cartesian target -> IK -> MuJoCo FK. It
    also asserts the commanded joint angles stay inside the Go1's joint ranges,
    because if they do not the position actuators clip at their ctrlrange and the
    executed gait quietly stops matching the commanded one.
    """
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    osc = cpg.CPG()
    cart = cpg.CartesianConfig()
    mu = np.ones(4)
    omega = np.full(4, 2.0 * np.pi * 2.0)  # 2 Hz

    lo, hi = model.jnt_range[1:].T
    worst_pos, worst_margin = 0.0, np.inf
    for _ in range(int(seconds / dt)):
        osc.step(dt, mu, omega)
        targets = osc.foot_targets(cart)
        q = legs_ik(targets)

        worst_margin = min(worst_margin, float(np.min([q - lo, hi - q])))

        data.qpos[7:] = q
        mujoco.mj_forward(model, data)
        for i, leg in enumerate(LEGS):
            achieved = foot_in_hip_frame(model, data, leg)
            worst_pos = max(worst_pos, float(np.abs(achieved - targets[i]).max()))

    print(f"  CPG targets: worst foot position error {worst_pos:.2e} m, "
          f"tightest joint-limit margin {worst_margin:.3f} rad")
    assert worst_pos < 1e-9, f"IK does not reach the CPG's targets ({worst_pos:.3e} m)"
    assert worst_margin > 0.0, "CPG commands joint angles outside the Go1's limits"


if __name__ == "__main__":
    test_ik_round_trip()
    test_cpg_targets_are_reachable()
    print("PASS")
