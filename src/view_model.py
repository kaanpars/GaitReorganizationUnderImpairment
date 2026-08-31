"""Open the Go1 in the MuJoCo viewer, standing at its nominal posture.

`python -m mujoco.viewer` also works, but drops the robot from the XML's
zero pose. This resets to the keyframe and holds it, which is what you want
when eyeballing whether a body condition is broken in the intended way
(CLAUDE.md: "verify it is broken in the intended way BEFORE training on it").

Usage:
    python src/view_model.py
    python src/view_model.py --paused          # step manually with the viewer
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from go1_model import DEFAULT_XML, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--keyframe", type=int, default=0, help="-1 for the raw XML pose")
    parser.add_argument("--paused", action="store_true", help="do not advance physics")
    args = parser.parse_args()

    model = load_model(args.xml)
    data = mujoco.MjData(model)

    if args.keyframe >= 0 and model.nkey > args.keyframe:
        # Also loads key_ctrl, so the position actuators hold the pose.
        mujoco.mj_resetDataKeyframe(model, data, args.keyframe)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if not args.paused:
                mujoco.mj_step(model, data)
            viewer.sync()
            # Run at wall-clock speed rather than as fast as the CPU allows.
            lag = model.opt.timestep - (time.time() - step_start)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
