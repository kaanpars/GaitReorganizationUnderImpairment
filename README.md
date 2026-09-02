# CPG-RL on Impaired Bodies

Learning CPG-based locomotion for a quadruped with a damaged leg — and comparing the
resulting gait adaptation to human gait compensation after limb loss.

**Status: No learning applied yet**

## Research question

When a legged robot's body is damaged, how does a CPG-RL controller reorganize its
gait — and does that reorganization resemble how human amputees compensate?

## Approach

1. Reproduce CPG-RL (Bellegarda & Ijspeert, RA-L 2022) on a Unitree Go1 in MuJoCo.
2. Define four body conditions: healthy, plus three impairments of the right hind leg.
3. Compare from-scratch, transfer, and multi-task conditioned policies.
4. Analyse the emergent gaits with metrics comparable to the clinical gait literature.

The policy modulates **CPG oscillator setpoints** (amplitude, frequency, coupling),
not raw joint torques. That indirection is the core of the method.

## Body conditions

| ID | Condition | Modification |
|---|---|---|
| `healthy` | baseline | stock Go1 |
| `weak` | reduced strength | torque limit → 40% |
| `stiff` | reduced ROM | knee range → 50% |
| `prosthetic` | passive limb | knee unactuated, passive spring-damper |

`prosthetic` is the headline condition — a crude analog of a transfemoral amputation
with a passive prosthesis.

## Metrics

Cost of transport · clinical symmetry index · inter-oscillator phase lags ·
push recovery · sample efficiency (steps to a fixed velocity-tracking threshold).

Episode reward is deliberately **not** reported as a result — it is an artifact of our
own reward design, not a property of the gait.

## Experimental protocol

- Every experiment runs ≥5 seeds; reported as median and interquartile range.
- Hyperparameters are tuned on the `healthy` body only, then frozen.
- Every run writes its full config to its run directory, so every number is traceable.
- Non-converged runs are reported, not rerun until they work.

## Stack

MuJoCo + Gymnasium · Stable-Baselines3 (PPO) · TensorBoard · NumPy / SciPy / Matplotlib

## Repo layout

```
src/cpg.py            coupled oscillators — pure NumPy, no MuJoCo dependency
src/go1_model.py      model loading, added sensors, analytic leg IK
src/inspect_model.py  week-1 diagnostic: does the Go1 support the method?
src/demo_trot.py      open-loop hand-tuned CPG trot (current milestone)
src/view_model.py     open the robot in the viewer at its nominal posture
tests/test_ik.py      IK verified against MuJoCo forward kinematics
```

Still to come: `src/envs/` (Gym env, body conditions, rewards), `src/train.py`,
`src/analysis/` (metrics, figures), `configs/`.

## What runs today

```bash
git clone --recurse-submodules https://github.com/kaanpars/GaitReorganizationUnderImpairment.git
cd GaitReorganizationUnderImpairment
pip install -r requirements.txt

python src/inspect_model.py     # checks every assumption the method needs
python tests/test_ik.py         # IK round-trip against MuJoCo FK
python src/demo_trot.py         # hand-tuned trot in the viewer
python src/demo_trot.py --gait bound --freq 3.0 --headless
```

## References

- Bellegarda & Ijspeert, *CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion*, IEEE RA-L 7(4), 2022.
- Bellegarda, Shafiee & Ijspeert, *Visual CPG-RL*, 2024.
- Ijspeert, *Central pattern generators for locomotion control in animals and robots: a review*, Neural Networks 21(4), 2008.
