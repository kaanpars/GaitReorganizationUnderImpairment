"""Inspect the Go1 MuJoCo model and check it supports the CPG-RL method.

This is a week-1 diagnostic, not part of the training loop. It answers one
question: does the stock Menagerie Go1 give us everything CPG-RL needs, and
everything the four impaired bodies need?

Every check below traces to a requirement from either
  Bellegarda & Ijspeert, "CPG-RL", RA-L 2022, or
  CLAUDE.md's spec for the four bodies / the five metrics.
Nothing else is reported.

Usage:
    python src/inspect_model.py
    python src/inspect_model.py --xml path/to/scene.xml --no-standing-test

Exit code is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from go1_model import DEFAULT_XML, LEG_KV, load_model

# Unitree convention. Index i here is oscillator index i in the CPG.
LEGS = ("FR", "FL", "RR", "RL")
JOINTS_PER_LEG = ("hip", "thigh", "calf")

# The leg every body condition in src/envs/bodies.py modifies.
IMPAIRED_LEG = "RR"

# Target policy rate. CPG-RL runs the policy well below the physics rate and
# integrates the oscillators on the physics clock.
CONTROL_HZ = 100.0


class Report:
    """Collects PASS / WARN / FAIL lines so the summary is honest about what failed."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self._add("PASS" if ok else "FAIL", name, detail)
        return ok

    def warn(self, name: str, detail: str = "") -> None:
        self._add("WARN", name, detail)

    def _add(self, status: str, name: str, detail: str) -> None:
        self.results.append((status, name))
        line = f"  [{status}] {name}"
        if detail:
            line += f"\n         {detail}"
        print(line)

    @property
    def failed(self) -> list[str]:
        return [n for s, n in self.results if s == "FAIL"]

    @property
    def warned(self) -> list[str]:
        return [n for s, n in self.results if s == "WARN"]


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def name_of(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i)


# --------------------------------------------------------------------------
# 1. Floating base and joint count
# --------------------------------------------------------------------------

def check_structure(model, rep: Report) -> None:
    section("1. Floating-base structure")

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
    print(f"  total mass = {model.body_mass.sum():.3f} kg   (denominator of cost of transport)")
    print(f"  gravity = {model.opt.gravity}")

    rep.check(len(free) == 1, "exactly one free joint (floating trunk)")
    rep.check(len(hinges) == 12, f"12 actuated hinge joints, found {len(hinges)}")
    rep.check(model.nq == 19 and model.nv == 18,
              "nq/nv match a 12-DoF quadruped on a free base (19/18)")


# --------------------------------------------------------------------------
# 2. Actuators: CPG-RL needs joint position control, not torque
# --------------------------------------------------------------------------

def check_actuators(model, rep: Report) -> dict[str, int]:
    section("2. Actuators (CPG-RL closes the loop with joint PD, not torques)")

    act_ids: dict[str, int] = {}
    expected = [f"{leg}_{j}" for leg in LEGS for j in JOINTS_PER_LEG]

    print(f"  kv = {LEG_KV} applied to every leg actuator by go1_model.load_model "
          "(the Menagerie XML itself ships kv = 0)")

    print(f"  {'idx':>3}  {'actuator':<10} {'joint':<16} {'kp':>6} {'kv':>5} "
          f"{'jnt damp':>8} {'forcerange':>16} {'ctrlrange':>18}")
    zero_kv = []
    for i in range(model.nu):
        aname = name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = model.actuator_trnid[i, 0]
        jname = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        # MuJoCo position actuator: gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv.
        kp = model.actuator_gainprm[i, 0]
        kv = -model.actuator_biasprm[i, 2] or 0.0  # avoid printing -0.0
        damp = model.dof_damping[model.jnt_dofadr[jid]]
        fr = model.actuator_forcerange[i]
        cr = model.actuator_ctrlrange[i]
        act_ids[aname] = i
        if kv == 0.0:
            zero_kv.append(aname)
        print(f"  {i:>3}  {aname:<10} {jname:<16} {kp:>6.1f} {kv:>5.1f} {damp:>8.1f} "
              f"[{fr[0]:>7.2f},{fr[1]:>6.2f}] [{cr[0]:>8.3f},{cr[1]:>7.3f}]")

    rep.check(all(model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT for i in range(model.nu)),
              "all actuators drive a joint directly (no tendons/gears to unpick)")
    rep.check(sorted(act_ids) == sorted(expected),
              "actuator names are the 12 expected LEG_joint names",
              f"missing: {sorted(set(expected) - set(act_ids))}" if sorted(act_ids) != sorted(expected) else "")
    rep.check([name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)] == expected,
              "actuator order is FR, FL, RR, RL x (hip, thigh, calf)",
              "CPG oscillator i maps to actuators 3i..3i+2")

    is_position = all(model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
                      and model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
                      for i in range(model.nu))
    rep.check(is_position, "actuators are position (PD) actuators, so CPG->IK->joint targets works")

    if zero_kv:
        rep.warn("position actuators have kv = 0 (no derivative term in the actuator)",
                 f"go1_model.load_model should have set kv = {LEG_KV} on every leg actuator - it did not "
                 f"reach {sorted(zero_kv)}. Until fixed, damping is only joint damping, which resists "
                 "absolute velocity rather than tracking error.")

    return act_ids


# --------------------------------------------------------------------------
# 3. Leg kinematics: the numbers analytic IK needs
# --------------------------------------------------------------------------

def check_kinematics(model, rep: Report) -> None:
    section("3. Leg kinematics (CPG emits foot positions; IK needs these link lengths)")

    lengths = {}
    for leg in LEGS:
        hip_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        thigh_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_thigh")
        calf_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        foot_s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)

        hip_off = model.body_pos[hip_b]          # trunk -> hip (abduction axis)
        abd_off = model.body_pos[thigh_b]        # hip -> thigh (lateral offset)
        thigh_len = abs(model.body_pos[calf_b][2])
        calf_len = abs(model.site_pos[foot_s][2])
        lengths[leg] = (float(np.linalg.norm(abd_off)), thigh_len, calf_len)

        print(f"  {leg}: trunk->hip {np.array2string(hip_off, precision=4)}  "
              f"hip->thigh {np.array2string(abd_off, precision=4)}  "
              f"l_thigh={thigh_len:.4f}  l_calf={calf_len:.4f}")

    shared = {tuple(round(x, 6) for x in v) for v in lengths.values()}
    l_abd, l_thigh, l_calf = next(iter(shared))

    rep.check(len(shared) == 1,
              "all four legs share one link-length set",
              f"l_abd={l_abd:.4f}  l_thigh={l_thigh:.4f}  l_calf={l_calf:.4f}  "
              "-> one analytic IK function serves every leg (sign flip on y for the left legs)")


# --------------------------------------------------------------------------
# 4. Feet: needed for stance detection (symmetry index, phase lags)
# --------------------------------------------------------------------------

def check_feet(model, rep: Report) -> None:
    section("4. Foot contacts (stance duration and step length come from these)")

    geoms, sites = {}, {}
    for leg in LEGS:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
        if g >= 0:
            geoms[leg] = g
            print(f"  {leg}: geom id {g} on body "
                  f"{name_of(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])}, "
                  f"condim={model.geom_condim[g]}, friction={model.geom_friction[g]}")
        if s >= 0:
            sites[leg] = s

    rep.check(len(geoms) == 4, "one named foot geom per leg (contact detection by geom id)")
    rep.check(len(sites) == 4, "one named foot site per leg (foot position without contact)")
    rep.check(all(model.geom_condim[g] >= 3 for g in geoms.values()),
              "foot contacts have friction (condim >= 3)")
    rep.check(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0,
              "a named floor geom exists to test contacts against")


# --------------------------------------------------------------------------
# 5. Nominal posture: the CPG's stance reference
# --------------------------------------------------------------------------

def check_nominal_posture(model, data, rep: Report) -> None:
    section("5. Nominal posture (CPG oscillates about a nominal foot position)")

    if not rep.check(model.nkey >= 1, "a keyframe exists to reset episodes from"):
        return

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    kname = name_of(model, mujoco.mjtObj.mjOBJ_KEY, 0)
    print(f"  keyframe '{kname}': base height {data.qpos[2]:.4f} m, "
          f"joint angles per leg {np.array2string(data.qpos[7:10], precision=3)}")

    for leg in LEGS:
        hip_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        foot_s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
        # Foot position expressed in the hip frame: what the CPG's output maps into.
        rel_world = data.site_xpos[foot_s] - data.xpos[hip_b]
        rel_hip = data.xmat[hip_b].reshape(3, 3).T @ rel_world
        print(f"  {leg}: nominal foot in hip frame = {np.array2string(rel_hip, precision=4)}")

    foot_z = [data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)][2] for leg in LEGS]
    rep.check(max(foot_z) - min(foot_z) < 1e-3, "all four feet level at the nominal posture")
    rep.check(0.15 < data.qpos[2] < 0.45, f"nominal base height {data.qpos[2]:.3f} m is a plausible stance")


# --------------------------------------------------------------------------
# 6. Observation sources
# --------------------------------------------------------------------------

def check_observations(model, rep: Report) -> None:
    section("6. Observation sources (CPG-RL observes base attitude, joint state, CPG state)")

    print(f"  nsensor = {model.nsensor}")
    if model.nsensor == 0:
        rep.warn("model defines no <sensor> elements",
                 "base orientation and angular velocity must be read from qpos[3:7] and qvel[3:6] "
                 "directly. Fine for this project - it only means no simulated IMU noise, which we "
                 "are not studying. The CPG state (r, theta, and their derivatives) comes from "
                 "src/cpg.py, not from the model.")
    imu = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    rep.check(imu >= 0, "an 'imu' site exists on the trunk if sensors are added later")


# --------------------------------------------------------------------------
# 7. Handles for the four body conditions
# --------------------------------------------------------------------------

def check_impairment_handles(model, rep: Report) -> None:
    section(f"7. Impairment handles on the {IMPAIRED_LEG} (right hind) leg")

    ok = True
    for jname in JOINTS_PER_LEG:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{IMPAIRED_LEG}_{jname}")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{IMPAIRED_LEG}_{jname}_joint")
        ok &= aid >= 0 and jid >= 0
        if aid < 0 or jid < 0:
            continue
        dof = model.jnt_dofadr[jid]
        print(f"  {IMPAIRED_LEG}_{jname}: actuator {aid}, joint {jid}, dof {dof}, "
              f"qpos {model.jnt_qposadr[jid]}")
        print(f"      forcerange {model.actuator_forcerange[aid]}  "
              f"jnt_range {model.jnt_range[jid]} (width {np.ptp(model.jnt_range[jid]):.3f} rad)  "
              f"stiffness {model.jnt_stiffness[jid]:.1f}  damping {model.dof_damping[dof]:.1f}")

    rep.check(bool(ok), f"all three {IMPAIRED_LEG} actuators and joints are addressable by name")

    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{IMPAIRED_LEG}_calf_joint")
    knee_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{IMPAIRED_LEG}_calf")

    rep.check(np.all(model.actuator_forcerange[knee_a] != 0),
              "'weak': actuator_forcerange is finite and scalable to 40%")
    rep.check(np.ptp(model.jnt_range[knee]) > 0 and model.jnt_limited[knee] == 1,
              "'stiff': knee jnt_range is limited and shrinkable to 50%",
              "shrink actuator_ctrlrange to match, or position targets will command outside the limit")
    rep.check(model.jnt_stiffness[knee] == 0.0,
              "'prosthetic': knee jnt_stiffness is currently 0, free to set as the passive spring",
              "the damper is dof_damping; the spring rest angle is qpos_spring. Note MuJoCo cannot "
              "delete an actuator at runtime - de-actuate the knee by zeroing its gainprm/biasprm "
              "and forcerange, and assert ctrl[8] has no effect in the body's verification test.")


# --------------------------------------------------------------------------
# 8. Timing
# --------------------------------------------------------------------------

def check_timing(model, rep: Report) -> None:
    section("8. Timing")

    dt = model.opt.timestep
    ratio = 1.0 / (CONTROL_HZ * dt)
    print(f"  physics timestep = {dt} s ({1 / dt:.0f} Hz)")
    print(f"  integrator = {mujoco.mjtIntegrator(model.opt.integrator).name}, "
          f"cone = {mujoco.mjtCone(model.opt.cone).name}, impratio = {model.opt.impratio:.0f}")
    print(f"  {CONTROL_HZ:.0f} Hz policy -> frame_skip = {ratio:.3f}")

    rep.check(abs(ratio - round(ratio)) < 1e-9,
              f"physics rate divides evenly into a {CONTROL_HZ:.0f} Hz policy rate",
              f"use frame_skip = {round(ratio)}; integrate the CPG at the physics rate inside the skip")


# --------------------------------------------------------------------------
# 9. Does it actually stand?
# --------------------------------------------------------------------------

def check_standing(model, data, rep: Report, seconds: float = 1.0) -> None:
    section("9. Standing test (holds the keyframe pose under its own PD gains)")

    if model.nkey < 1:
        rep.warn("no keyframe, standing test skipped")
        return

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    z0 = float(data.qpos[2])

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg): leg for leg in LEGS}

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)

    in_contact = set()
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        for g in (g1, g2):
            if g in foot_geoms and floor in (g1, g2):
                in_contact.add(foot_geoms[g])

    z1 = float(data.qpos[2])
    print(f"  base height {z0:.4f} -> {z1:.4f} m after {seconds:.1f} s (drop {z0 - z1:+.4f} m)")
    print(f"  feet on floor: {sorted(in_contact) or 'none'}")
    print(f"  peak |actuator force| = {np.abs(data.actuator_force).max():.2f} Nm")

    rep.check(abs(z1 - z0) < 0.03, "base height holds within 3 cm (PD gains support the robot)")
    rep.check(len(in_contact) == 4, f"all four feet in contact, found {len(in_contact)}")
    rep.check(np.isfinite(data.qpos).all(), "simulation is numerically stable (no NaNs)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MuJoCo scene to inspect")
    parser.add_argument("--no-standing-test", action="store_true",
                        help="skip the 1 s forward simulation")
    args = parser.parse_args()

    if not args.xml.exists():
        print(f"model not found: {args.xml}", file=sys.stderr)
        return 1

    print(f"model: {args.xml}")
    print(f"mujoco {mujoco.__version__}, numpy {np.__version__}")

    model = load_model(args.xml)
    data = mujoco.MjData(model)
    rep = Report()

    check_structure(model, rep)
    check_actuators(model, rep)
    check_kinematics(model, rep)
    check_feet(model, rep)
    check_nominal_posture(model, data, rep)
    check_observations(model, rep)
    check_impairment_handles(model, rep)
    check_timing(model, rep)
    if not args.no_standing_test:
        check_standing(model, data, rep)

    section("Summary")
    total = len(rep.results)
    print(f"  {total - len(rep.failed) - len(rep.warned)} passed, "
          f"{len(rep.warned)} warnings, {len(rep.failed)} failed")
    for n in rep.warned:
        print(f"  WARN: {n}")
    for n in rep.failed:
        print(f"  FAIL: {n}")
    if not rep.failed:
        print("\n  Model is suitable for CPG-RL and for the four body conditions.")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
