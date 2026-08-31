"""Coupled amplitude-phase oscillators - one per leg.

Implements the CPG of Bellegarda & Ijspeert, "CPG-RL", RA-L 2022, Sec. III-A.
Each leg i carries an amplitude r_i and a phase theta_i:

    r_ddot_i = a * ( (a/4) * (mu_i - r_i) - r_dot_i )
    theta_dot_i = omega_i + sum_j r_j * w_ij * sin(theta_j - theta_i - phi_ij)

The amplitude equation is a critically damped second-order system, so r_i
converges to its setpoint mu_i without overshoot; `a` sets how fast. The phase
equation is a Kuramoto coupling that pulls the legs toward the phase offsets
phi_ij, which is what makes a gait rather than four independent legs.

The RL policy modulates the SETPOINTS (mu, omega, and later the coupling), never
the joint torques. That indirection is the method. See CLAUDE.md.

This module deliberately imports nothing but numpy: no MuJoCo, no gymnasium. It
must stay that way so the oscillators can be unit-tested on their own and later
cross-checked against an embedded implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_LEGS = 4

# Leg order is Unitree's, matching the actuator order in the Go1 model:
# oscillator i drives actuators 3i..3i+2.
LEGS = ("FR", "FL", "RR", "RL")

# Nominal phase of each leg within the cycle, in fractions of 2*pi. The coupling
# matrix phi_ij is built from these as phi_ij = 2*pi*(f_j - f_i).
GAITS: dict[str, tuple[float, float, float, float]] = {
    #        FR     FL     RR     RL
    "trot":  (0.00, 0.50, 0.50, 0.00),   # diagonal pairs together
    "pace":  (0.00, 0.50, 0.00, 0.50),   # lateral pairs together
    "bound": (0.00, 0.00, 0.50, 0.50),   # front pair, then hind pair
    "walk":  (0.00, 0.50, 0.75, 0.25),   # one foot at a time: FR, RL, FL, RR
}


@dataclass
class CPGConfig:
    """Oscillator constants. Hand-tuned in week 2; frozen before any training."""

    # Amplitude convergence factor `a`. Larger = r tracks mu faster.
    convergence: float = 50.0
    # Uniform coupling strength w_ij between every pair of oscillators.
    coupling: float = 1.0
    gait: str = "trot"
    # Amplitudes are clipped to stay in the range the Cartesian mapping expects.
    mu_range: tuple[float, float] = (0.0, 2.0)


@dataclass
class CartesianConfig:
    """Maps oscillator state to a foot position in the hip frame (paper Sec. III-A).

    Defaults are hand-tuned for the Go1; `stand_height` and `hip_offset_y` come
    from the model itself (see src/inspect_model.py section 5), the rest are the
    three knobs actually tuned in week 2.
    """

    stand_height: float = 0.2648      # h: nominal foot depth below the hip
    step_length: float = 0.060        # d_step: half-stride at r = 1
    ground_clearance: float = 0.050   # g_c: peak foot lift during swing
    ground_penetration: float = 0.010 # g_p: how far below h the foot presses in stance
    hip_offset_y: float = 0.080       # lateral thigh offset; sign applied per leg
    # +1 for legs whose thigh sits at +y (left), -1 for right. FR, FL, RR, RL.
    y_sign: tuple[int, int, int, int] = (-1, 1, -1, 1)


def phase_bias_matrix(gait: str) -> np.ndarray:
    """phi_ij = desired phase of leg j minus leg i, in radians."""
    if gait not in GAITS:
        raise ValueError(f"unknown gait {gait!r}; known gaits: {sorted(GAITS)}")
    f = np.asarray(GAITS[gait], dtype=float)
    return 2.0 * np.pi * (f[None, :] - f[:, None])


class CPG:
    """Four coupled oscillators, integrated with explicit Euler."""

    def __init__(self, config: CPGConfig | None = None, seed: int | None = None):
        self.config = config or CPGConfig()
        self.phi = phase_bias_matrix(self.config.gait)
        self.w = self.config.coupling * (1.0 - np.eye(N_LEGS))  # no self-coupling
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, r: np.ndarray | None = None, theta: np.ndarray | None = None) -> None:
        """Reset to the gait's nominal phases with a small amplitude.

        r starts near zero rather than at mu so the first strides grow in from a
        stand instead of the feet snapping to full stride on step 0.
        """
        f = np.asarray(GAITS[self.config.gait], dtype=float)
        self.theta = (2.0 * np.pi * f).copy() if theta is None else np.asarray(theta, float).copy()
        self.r = np.full(N_LEGS, 0.0) if r is None else np.asarray(r, float).copy()
        self.r_dot = np.zeros(N_LEGS)

    def step(self, dt: float, mu: np.ndarray, omega: np.ndarray) -> None:
        """Advance the oscillators by dt.

        mu:    amplitude setpoint per leg (policy output, or a constant in week 2)
        omega: intrinsic frequency per leg in rad/s (policy output)
        """
        mu = np.clip(np.asarray(mu, float), *self.config.mu_range)
        omega = np.asarray(omega, float)

        a = self.config.convergence
        r_ddot = a * ((a / 4.0) * (mu - self.r) - self.r_dot)

        # theta_j - theta_i - phi_ij, summed over j, weighted by r_j and w_ij.
        diff = self.theta[None, :] - self.theta[:, None] - self.phi
        theta_dot = omega + (self.r[None, :] * self.w * np.sin(diff)).sum(axis=1)

        self.r_dot += r_ddot * dt
        self.r += self.r_dot * dt
        self.theta += theta_dot * dt
        self.theta = np.mod(self.theta, 2.0 * np.pi)

    def foot_targets(self, cart: CartesianConfig) -> np.ndarray:
        """Oscillator state -> desired foot position per leg, in that leg's hip frame.

        Returns (4, 3): x forward, y lateral, z up (negative, foot below hip).

        The phase sets swing vs. stance: sin(theta) > 0 is swing, where the foot
        lifts by g_c; otherwise it is stance, pressing g_p into the ground so the
        contact stays loaded.
        """
        x = -cart.step_length * self.r * np.cos(self.theta)
        y = cart.hip_offset_y * np.asarray(cart.y_sign, dtype=float)

        sin_t = np.sin(self.theta)
        swing = sin_t > 0.0
        g = np.where(swing, cart.ground_clearance, cart.ground_penetration)
        z = -cart.stand_height + g * sin_t

        return np.stack([x, y, z], axis=1)

    @property
    def state(self) -> dict[str, np.ndarray]:
        """Everything the env should put in the info dict every step.

        theta is the readout used to identify which gait emerged (CLAUDE.md,
        "Metrics: phase lags"), so it must survive to the analysis stage.
        """
        return {"r": self.r.copy(), "r_dot": self.r_dot.copy(), "theta": self.theta.copy()}

    def phase_lags(self, reference: int = 0) -> np.ndarray:
        """Phase of each leg relative to `reference`, wrapped to [0, 2*pi).

        This is the gait signature: ~(0, pi, pi, 0) is a trot, ~(0, 0, pi, pi) a
        bound, and so on.
        """
        return np.mod(self.theta - self.theta[reference], 2.0 * np.pi)
