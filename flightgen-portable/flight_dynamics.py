"""
Flight Dynamics Simulator - Portable Version (CPU-only)

6-DOF flight dynamics with NumPy. Stripped of GPU/CuPy/PyTorch dependencies
for maximum portability.

Original: AIDA/gpu-flight-dynamics/python/flight_dynamics.py
Author: Kushal Koirala
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from enum import IntEnum


class StateIndex(IntEnum):
    X = 0       # Position North [m]
    Y = 1       # Position East [m]
    Z = 2       # Position Down [m]
    U = 3       # Body velocity X [m/s]
    V = 4       # Body velocity Y [m/s]
    W = 5       # Body velocity Z [m/s]
    PHI = 6     # Roll [rad]
    THETA = 7   # Pitch [rad]
    PSI = 8     # Yaw [rad]
    P = 9       # Roll rate [rad/s]
    Q = 10      # Pitch rate [rad/s]
    R = 11      # Yaw rate [rad/s]


class ControlIndex(IntEnum):
    THROTTLE = 0
    AILERON = 1
    ELEVATOR = 2
    RUDDER = 3
    FLAP = 4
    SPOILER = 5
    BRAKE = 6


STATE_DIM = 12
CONTROL_DIM = 7


@dataclass
class MassProperties:
    mass: float = 1043.0
    Ixx: float = 1285.0
    Iyy: float = 1825.0
    Izz: float = 2667.0
    Ixz: float = 0.0


@dataclass
class Geometry:
    S: float = 16.17
    b: float = 10.92
    c: float = 1.49
    e: float = 0.8

    @property
    def AR(self) -> float:
        return self.b ** 2 / self.S


@dataclass
class LongitudinalDerivatives:
    CL0: float = 0.307
    CLa: float = 4.41
    CLq: float = 3.9
    CLde: float = 0.43
    CLmax: float = 1.4
    CLmin: float = -0.5
    CD0: float = 0.027
    K: float = 0.054
    CDa: float = 0.0
    Cm0: float = 0.04
    Cma: float = -0.613
    Cmq: float = -12.4
    Cmde: float = -1.122
    dCL_flap: float = 0.5
    dCD_flap: float = 0.08
    dCm_flap: float = -0.12
    dCL_spoiler: float = -0.4
    dCD_spoiler: float = 0.10


@dataclass
class LateralDerivatives:
    CYb: float = -0.393
    CYp: float = -0.075
    CYr: float = 0.214
    CYda: float = 0.0
    CYdr: float = 0.187
    Clb: float = -0.0923
    Clp: float = -0.484
    Clr: float = 0.0798
    Clda: float = 0.229
    Cldr: float = 0.0147
    Cnb: float = 0.0587
    Cnp: float = -0.0278
    Cnr: float = -0.0937
    Cnda: float = -0.0216
    Cndr: float = -0.0645


@dataclass
class PropulsionParams:
    thrust_max: float = 2500.0
    thrust_min: float = 50.0
    tau: float = 0.5


@dataclass
class AircraftParams:
    mass: MassProperties = field(default_factory=MassProperties)
    geom: Geometry = field(default_factory=Geometry)
    longi: LongitudinalDerivatives = field(default_factory=LongitudinalDerivatives)
    latdi: LateralDerivatives = field(default_factory=LateralDerivatives)
    prop: PropulsionParams = field(default_factory=PropulsionParams)


class FlightSimulator:
    """CPU-only 6-DOF flight dynamics simulator."""

    def __init__(self, n_instances=1, params=None, dt=0.01, use_gpu=False, integration_method=2):
        self.n_instances = n_instances
        self.params = params or AircraftParams()
        self.dt = dt
        self.integration_method = integration_method
        self.xp = np

        self.states = np.zeros((n_instances, STATE_DIM), dtype=np.float32)
        self.controls = np.zeros((n_instances, CONTROL_DIM), dtype=np.float32)
        self.time = 0.0
        self._init_trim()
        print(f"FlightSimulator: {n_instances} instances on CPU (NumPy)")

    def _init_trim(self):
        self.states[:, StateIndex.Z] = -1000.0
        self.states[:, StateIndex.U] = 50.0
        self.controls[:, ControlIndex.THROTTLE] = 0.4

    def reset(self, initial_state=None, initial_control=None):
        self.time = 0.0
        if initial_state is not None:
            state = np.asarray(initial_state, dtype=np.float32)
            if state.ndim == 1:
                self.states[:] = state
            else:
                self.states[:] = state
        else:
            self.states[:] = 0.0
            self._init_trim()

        if initial_control is not None:
            ctrl = np.asarray(initial_control, dtype=np.float32)
            if ctrl.ndim == 1:
                self.controls[:] = ctrl
            else:
                self.controls[:] = ctrl
        else:
            self.controls[:] = 0.0
            self.controls[:, ControlIndex.THROTTLE] = 0.4

    def set_controls(self, controls):
        ctrl = np.asarray(controls, dtype=np.float32)
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(1, -1)
        if ctrl.shape[-1] == 4:
            self.controls[:, :4] = ctrl
        elif ctrl.shape[-1] >= 6:
            self.controls[:] = ctrl[:, :CONTROL_DIM]
        else:
            raise ValueError(f"Controls must have 4 or 6+ elements, got {ctrl.shape[-1]}")

    def set_flaps(self, flap_setting):
        self.controls[:, ControlIndex.FLAP] = np.clip(flap_setting, 0.0, 1.0)

    def set_spoilers(self, spoiler_setting):
        self.controls[:, ControlIndex.SPOILER] = np.clip(spoiler_setting, 0.0, 1.0)

    def step(self):
        if self.integration_method == 0:
            self._step_euler()
        elif self.integration_method == 1:
            self._step_rk2()
        else:
            self._step_rk4()
        self.time += self.dt

    def step_n(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def _compute_derivatives(self, states):
        z = states[:, StateIndex.Z]
        u = states[:, StateIndex.U]
        v = states[:, StateIndex.V]
        w = states[:, StateIndex.W]
        phi = states[:, StateIndex.PHI]
        theta = states[:, StateIndex.THETA]
        psi = states[:, StateIndex.PSI]
        p = states[:, StateIndex.P]
        q = states[:, StateIndex.Q]
        r = states[:, StateIndex.R]

        throttle = self.controls[:, ControlIndex.THROTTLE]
        da = self.controls[:, ControlIndex.AILERON]
        de = self.controls[:, ControlIndex.ELEVATOR]
        dr = self.controls[:, ControlIndex.RUDDER]
        flap = np.clip(self.controls[:, ControlIndex.FLAP], 0.0, 1.0)
        spoiler = np.clip(self.controls[:, ControlIndex.SPOILER], 0.0, 1.0)

        altitude = np.maximum(0.0, -z)
        T = 288.15 - 0.0065 * altitude
        rho = 1.225 * np.power(T / 288.15, 4.256)

        V = np.sqrt(u**2 + v**2 + w**2)
        V = np.maximum(V, 0.1)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V, -1, 1))

        qbar = 0.5 * rho * V**2

        phat = p * self.params.geom.b / (2 * V)
        qhat = q * self.params.geom.c / (2 * V)
        rhat = r * self.params.geom.b / (2 * V)

        longi = self.params.longi
        latdi = self.params.latdi

        CL = longi.CL0 + longi.CLa * alpha + longi.CLq * qhat + longi.CLde * de + longi.dCL_flap * flap + longi.dCL_spoiler * spoiler
        CD = longi.CD0 + longi.K * CL**2 + longi.dCD_flap * flap + longi.dCD_spoiler * spoiler
        Cm = longi.Cm0 + longi.Cma * alpha + longi.Cmq * qhat + longi.Cmde * de + longi.dCm_flap * flap

        CY = latdi.CYb * beta + latdi.CYp * phat + latdi.CYr * rhat + latdi.CYdr * dr
        Cl = latdi.Clb * beta + latdi.Clp * phat + latdi.Clr * rhat + latdi.Clda * da
        Cn = latdi.Cnb * beta + latdi.Cnp * phat + latdi.Cnr * rhat + latdi.Cndr * dr

        S = self.params.geom.S
        L_aero = qbar * S * CL
        D_aero = qbar * S * CD
        Y_aero = qbar * S * CY

        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        Fx = -D_aero * cos_a + L_aero * sin_a + throttle * self.params.prop.thrust_max
        Fy = Y_aero
        Fz = -D_aero * sin_a - L_aero * cos_a

        b = self.params.geom.b
        c = self.params.geom.c
        L_mom = qbar * S * b * Cl
        M_mom = qbar * S * c * Cm
        N_mom = qbar * S * b * Cn

        g = 9.80665
        gx = -g * np.sin(theta)
        gy = g * np.cos(theta) * np.sin(phi)
        gz = g * np.cos(theta) * np.cos(phi)

        mass = self.params.mass.mass
        u_dot = Fx / mass - q*w + r*v + gx
        v_dot = Fy / mass - r*u + p*w + gy
        w_dot = Fz / mass - p*v + q*u + gz

        Ixx = self.params.mass.Ixx
        Iyy = self.params.mass.Iyy
        Izz = self.params.mass.Izz

        p_dot = L_mom / Ixx
        q_dot = M_mom / Iyy
        r_dot = N_mom / Izz

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)

        phi_dot = p + (q * sin_phi + r * cos_phi) * tan_theta
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = (q * sin_phi + r * cos_phi) / np.maximum(np.abs(cos_theta), 0.001)

        sin_theta = np.sin(theta)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        R11 = cos_theta * cos_psi
        R12 = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
        R13 = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
        R21 = cos_theta * sin_psi
        R22 = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
        R23 = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
        R31 = -sin_theta
        R32 = sin_phi * cos_theta
        R33 = cos_phi * cos_theta

        x_dot = R11*u + R12*v + R13*w
        y_dot = R21*u + R22*v + R23*w
        z_dot = R31*u + R32*v + R33*w

        state_dot = np.zeros_like(states)
        state_dot[:, StateIndex.X] = x_dot
        state_dot[:, StateIndex.Y] = y_dot
        state_dot[:, StateIndex.Z] = z_dot
        state_dot[:, StateIndex.U] = u_dot
        state_dot[:, StateIndex.V] = v_dot
        state_dot[:, StateIndex.W] = w_dot
        state_dot[:, StateIndex.PHI] = phi_dot
        state_dot[:, StateIndex.THETA] = theta_dot
        state_dot[:, StateIndex.PSI] = psi_dot
        state_dot[:, StateIndex.P] = p_dot
        state_dot[:, StateIndex.Q] = q_dot
        state_dot[:, StateIndex.R] = r_dot

        return state_dot

    def _step_euler(self):
        state_dot = self._compute_derivatives(self.states)
        self.states += self.dt * state_dot
        self._normalize_angles()
        self._enforce_ground_contact()

    def _step_rk2(self):
        k1 = self._compute_derivatives(self.states)
        k2 = self._compute_derivatives(self.states + self.dt * k1)
        self.states += 0.5 * self.dt * (k1 + k2)
        self._normalize_angles()
        self._enforce_ground_contact()

    def _step_rk4(self):
        dt = self.dt
        k1 = self._compute_derivatives(self.states)
        k2 = self._compute_derivatives(self.states + 0.5 * dt * k1)
        k3 = self._compute_derivatives(self.states + 0.5 * dt * k2)
        k4 = self._compute_derivatives(self.states + dt * k3)
        self.states += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self._normalize_angles()
        self._enforce_ground_contact()

    def _normalize_angles(self):
        pi = np.pi
        for idx in [StateIndex.PHI, StateIndex.THETA, StateIndex.PSI]:
            self.states[:, idx] = np.mod(self.states[:, idx] + pi, 2*pi) - pi

    def _enforce_ground_contact(self):
        z = self.states[:, StateIndex.Z]
        w = self.states[:, StateIndex.W]
        theta = self.states[:, StateIndex.THETA]

        on_ground = z >= 0.0
        if np.any(on_ground):
            self.states[:, StateIndex.Z] = np.minimum(z, 0.0)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            u = self.states[:, StateIndex.U]
            z_dot_approx = -sin_theta * u + cos_theta * w
            going_down = z_dot_approx > 0
            needs_correction = on_ground & going_down

            if np.any(needs_correction):
                self.states[:, StateIndex.W] = np.where(needs_correction, 0.0, w)
                max_pitch_down = -0.087
                self.states[:, StateIndex.THETA] = np.where(
                    needs_correction & (theta < max_pitch_down), max_pitch_down, theta
                )

            brake_input = self.controls[:, ControlIndex.BRAKE]
            u = self.states[:, StateIndex.U]
            moving_forward = u > 1.0
            apply_brakes = on_ground & moving_forward & (brake_input > 0.01)
            if np.any(apply_brakes):
                max_brake_decel = 3.0 * self.dt
                decel = brake_input * max_brake_decel
                new_u = np.maximum(u - decel, 0.0)
                self.states[:, StateIndex.U] = np.where(apply_brakes, new_u, u)

    def get_states(self):
        return self.states.copy()

    def get_altitude(self):
        return -self.states[:, StateIndex.Z].copy()

    def get_airspeed(self):
        u = self.states[:, StateIndex.U]
        v = self.states[:, StateIndex.V]
        w = self.states[:, StateIndex.W]
        return np.sqrt(u**2 + v**2 + w**2)
