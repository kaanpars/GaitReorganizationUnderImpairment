"""Load the Go1 model the way every part of this project should load it.

The stock MuJoCo Menagerie Go1 (`mujoco_menagerie/unitree_go1/scene.xml`) is left
untouched on disk. Two things it lacks for CPG-RL are added here, in one place:

  1. A <sensor> block. The Menagerie XML defines none, so `model.nsensor == 0` and
     the observation has to be scraped out of `data.qpos`/`data.qvel`. We add a
     named trunk IMU (orientation, angular + linear velocity, acceleration) and
     joint pos/vel sensors for the 12 hinges.
  2. A derivative gain (kv) on the 12 leg position actuators. The Menagerie XML
     ships them as <position kp="100" .../> with no kv, leaning on <joint damping>.
     Joint damping resists absolute joint velocity, not tracking-error velocity -
     fine while the target is held still, but CPG-RL commands a continuously moving
     joint target. kp=100 / kd~2 matches the joint PD in Bellegarda & Ijspeert,
     CPG-RL 2022.

Sensors are added through MuJoCo's mjSpec API rather than an <include> wrapper XML
because a wrapper living outside `mujoco_menagerie/` breaks that model's relative
mesh paths. This module is also where src/envs/bodies.py will hook its runtime
model edits later (e.g. de-actuating the prosthetic knee).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

# Unitree order: oscillator i in the CPG maps to actuators/sensors 3i..3i+2.
LEGS = ("FR", "FL", "RR", "RL")
JOINTS_PER_LEG = ("hip", "thigh", "calf")

DEFAULT_XML = (
    Path(__file__).resolve().parents[1] / "mujoco_menagerie" / "unitree_go1" / "scene.xml"
)

# Derivative gain applied to all 12 leg position actuators. Joint damping in the
# Menagerie XML is left untouched: at steady stance (q_dot ~ 0) this term is ~0, so
# the standing pose is unaffected; during motion it only adds damping, which is
# stable. If the standing test ever regresses, lower this toward 0 - do not touch
# joint damping.
LEG_KV = 2.0

# name -> (sensor type, referenced object type, object name). The IMU sensors ride
# the "imu" site already on the trunk.
_TRUNK_SENSORS = {
    "trunk_quat": (mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_SITE, "imu"),
    "trunk_gyro": (mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE, "imu"),
    "trunk_acc": (mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtObj.mjOBJ_SITE, "imu"),
    "trunk_linvel": (mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtObj.mjOBJ_SITE, "imu"),
}


def _add_sensors(spec: mujoco.MjSpec) -> None:
    def add(name: str, stype, objtype, objname: str) -> None:
        s = spec.add_sensor()
        s.name = name
        s.type = stype
        s.objtype = objtype
        s.objname = objname

    for name, (stype, objtype, objname) in _TRUNK_SENSORS.items():
        add(name, stype, objtype, objname)
    for leg in LEGS:
        for j in JOINTS_PER_LEG:
            add(f"{leg}_{j}_pos", mujoco.mjtSensor.mjSENS_JOINTPOS,
                mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")
    for leg in LEGS:
        for j in JOINTS_PER_LEG:
            add(f"{leg}_{j}_vel", mujoco.mjtSensor.mjSENS_JOINTVEL,
                mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")


def load_model(xml_path: Path | str = DEFAULT_XML, kv: float = LEG_KV) -> mujoco.MjModel:
    """Compile the Menagerie Go1 scene with sensors added and the leg-actuator kv set."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    _add_sensors(spec)
    model = spec.compile()
    # MuJoCo position actuator: biasprm = [0, -kp, -kv]. Every Go1 leg actuator is a
    # position actuator, so this covers all of them.
    model.actuator_biasprm[:, 2] = -kv
    return model


# --- Leg kinematics -------------------------------------------------------
#
# The CPG emits a desired foot position; the joint PD tracks a desired angle.
# This is the map between them. Link lengths are read off the model by
# src/inspect_model.py section 3 and are identical on all four legs:
#
#   L_ABDUCTION  hip -> thigh lateral offset, sign per leg
#   L_THIGH      thigh -> knee
#   L_CALF       knee -> foot
#
# Frame convention (hip frame, from the Menagerie XML): the abduction joint
# rotates about x, the thigh and calf joints about y, x is forward, z is up, so
# a foot below the hip has negative z.

L_ABDUCTION = 0.080
L_THIGH = 0.213
L_CALF = 0.213

# +1 for legs whose thigh sits at +y (left), -1 for right. FR, FL, RR, RL.
Y_SIGN = (-1, 1, -1, 1)

# Thigh joint range from the model. It exceeds pi at the top, so a solution
# wrapped to [-pi, pi] can land outside the joint's reachable interval while the
# same angle plus 2*pi lands inside. The abduction and calf ranges are both well
# inside [-pi, pi] and need no such care.
Q_THIGH_RANGE = (-0.686, 4.501)


def foot_ik(target: "np.ndarray", y_sign: int) -> "np.ndarray":
    """Foot position in the hip frame -> (abduction, thigh, calf) joint angles.

    Analytic and exact, but single-branch. A foot position does not determine the
    leg configuration on its own: the same point is reachable with the leg
    extended downward and, at a different abduction angle, with it folded up over
    the hip. This returns the downward branch, which is the only one the CPG's
    Cartesian mapping ever asks for (commanded z stays near -0.22 to -0.28 m).

    Targets outside the leg's reach are clamped onto the workspace boundary
    rather than raising, so a policy exploring a bad amplitude degrades smoothly
    instead of killing the episode with an exception.

    Verified against MuJoCo forward kinematics to 3e-14 rad in tests/test_ik.py.
    """
    import numpy as np

    x, y, z = float(target[0]), float(target[1]), float(target[2])
    l1 = y_sign * L_ABDUCTION

    # Abduction: in the y-z plane the foot lies on a circle of radius
    # sqrt(l1^2 + c^2) about the hip, where c is the in-plane leg extension.
    yz_sq = y * y + z * z
    c = -np.sqrt(max(yz_sq - l1 * l1, 0.0))  # negative: the foot is below the hip
    q_abd = np.arctan2(z, y) - np.arctan2(c, l1)
    q_abd = np.arctan2(np.sin(q_abd), np.cos(q_abd))  # wrap to [-pi, pi]

    # Knee: planar two-link, with the target clamped to the annulus the leg can reach.
    reach = np.hypot(x, c)
    reach = np.clip(reach, abs(L_THIGH - L_CALF) + 1e-6, L_THIGH + L_CALF - 1e-6)
    cos_knee = (reach * reach - L_THIGH**2 - L_CALF**2) / (2.0 * L_THIGH * L_CALF)
    q_calf = -np.arccos(np.clip(cos_knee, -1.0, 1.0))

    # Thigh: rotate the two-link solution to point at the target.
    a_len = L_THIGH + L_CALF * np.cos(q_calf)
    b_len = L_CALF * np.sin(q_calf)
    q_thigh = np.arctan2(-x, -c) - np.arctan2(b_len, a_len)
    q_thigh = np.arctan2(np.sin(q_thigh), np.cos(q_thigh))
    if q_thigh < Q_THIGH_RANGE[0] and q_thigh + 2 * np.pi <= Q_THIGH_RANGE[1]:
        q_thigh += 2 * np.pi  # the other branch is the one inside the joint range

    return np.array([q_abd, q_thigh, q_calf])


def legs_ik(targets: "np.ndarray") -> "np.ndarray":
    """(4, 3) foot targets in hip frames -> (12,) joint angles in actuator order."""
    import numpy as np

    return np.concatenate([foot_ik(targets[i], Y_SIGN[i]) for i in range(4)])
