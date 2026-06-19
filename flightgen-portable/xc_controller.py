"""
Generalized Cross-Country Controller with Triangle Intercept Pattern - Portable Version

Original: AIDA/scripts/generalized_xc_controller.py
Author: Kushal Koirala (with Claude Code)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from flight_dynamics import StateIndex, CONTROL_DIM

# Unit conversion constants
FT_TO_M = 0.3048
M_TO_FT = 3.28084
NM_TO_FT = 6076.12
FT_TO_NM = 1 / 6076.12
KTS_TO_FPS = 1.68781
FPS_TO_KTS = 1 / 1.68781


class XCPhase(Enum):
    GROUND_ROLL = 1
    ROTATION = 2
    INITIAL_CLIMB = 3
    CLIMB = 4
    CRUISE_TO_TP = 5
    TURN_TO_INTERCEPT = 6
    INTERCEPT_LEG = 7
    FINAL_APPROACH = 8
    SHORT_FINAL = 9
    LANDING = 10
    LANDED = 11


@dataclass
class AirportConfig:
    icao: str
    name: str
    x_ft: float = 0.0
    y_ft: float = 0.0
    elevation_ft: float = 0.0
    runway_heading_deg: float = 0.0
    viewer_runway_heading_deg: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    def get_viewer_heading_deg(self) -> float:
        if self.viewer_runway_heading_deg is not None:
            return self.viewer_runway_heading_deg
        return self.runway_heading_deg


KANSAS_AIRPORTS = {
    "SN65": AirportConfig(
        icao="SN65", name="Lake Waltanna Airport",
        x_ft=0.0, y_ft=0.0, elevation_ft=1448.0,
        runway_heading_deg=4.0, lat=37.594167, lon=-97.615833
    ),
    "KHUT": AirportConfig(
        icao="KHUT", name="Hutchinson Regional Airport",
        x_ft=52800.0 * M_TO_FT, y_ft=-21300.0 * M_TO_FT,
        elevation_ft=1542.0, runway_heading_deg=314.0,
        lat=38.066167, lon=-97.860500
    ),
    "KICT": AirportConfig(
        icao="KICT", name="Wichita Eisenhower National",
        x_ft=6156.0 * M_TO_FT, y_ft=15000.0 * M_TO_FT,
        elevation_ft=1333.0, runway_heading_deg=14.0,
        lat=37.649944, lon=-97.433056
    ),
    "KAAO": AirportConfig(
        icao="KAAO", name="Colonel James Jabara Airport",
        x_ft=55880.0, y_ft=115140.0, elevation_ft=1421.0,
        runway_heading_deg=180.0, lat=37.747500, lon=-97.221389
    ),
    "K50K": AirportConfig(
        icao="K50K", name="Pawnee Municipal Airport",
        x_ft=215034.0, y_ft=-436524.0, elevation_ft=2200.0,
        runway_heading_deg=170.0, lat=38.184, lon=-99.127
    ),
}


def latlon_to_xy(lat, lon, ref_lat=37.594167, ref_lon=-97.615833):
    lat_to_ft = 364567.0
    lon_to_ft = 364567.0 * np.cos(np.deg2rad(ref_lat))
    x = (lat - ref_lat) * lat_to_ft
    y = (lon - ref_lon) * lon_to_ft
    return x, y


class GeneralizedXCController:
    def __init__(self, origin, destination, cruise_altitude_ft=5500.0,
                 pattern_altitude_ft=1500.0, tp_distance_nm=10.0):
        self.origin = origin
        self.destination = destination
        self.cruise_altitude_ft = cruise_altitude_ft
        self.pattern_altitude_ft = pattern_altitude_ft

        self.v_rotate = 54.0 * KTS_TO_FPS
        self.v_climb = 74.0 * KTS_TO_FPS
        self.v_cruise = 110.0 * KTS_TO_FPS
        self.v_approach = 65.0 * KTS_TO_FPS
        self.v_touchdown = 50.0 * KTS_TO_FPS

        self.initial_climb_alt_ft = 100.0
        self.turn_to_cruise_altitude_ft = 1000.0
        self.flare_altitude_ft = 50.0
        self.touchdown_altitude_ft = 3.0
        self.short_final_distance_ft = 2500.0

        self.departure_heading = np.deg2rad(origin.runway_heading_deg)
        self.runway_heading = np.deg2rad(destination.runway_heading_deg)
        self.backcourse = np.deg2rad((destination.runway_heading_deg + 180) % 360)

        self.dest_x_ft = destination.x_ft
        self.dest_y_ft = destination.y_ft

        threshold_offset_ft = 610.0 * M_TO_FT
        self.threshold_x_ft = self.dest_x_ft + threshold_offset_ft * np.cos(self.backcourse)
        self.threshold_y_ft = self.dest_y_ft + threshold_offset_ft * np.sin(self.backcourse)

        self.aimpoint_x_ft = self.threshold_x_ft
        self.aimpoint_y_ft = self.threshold_y_ft

        tp_distance_ft = tp_distance_nm * NM_TO_FT
        self.tp_x_ft = self.threshold_x_ft + tp_distance_ft * np.cos(self.backcourse)
        self.tp_y_ft = self.threshold_y_ft + tp_distance_ft * np.sin(self.backcourse)

        self.cruise_heading = np.arctan2(self.tp_y_ft - origin.y_ft,
                                          self.tp_x_ft - origin.x_ft)

        self.tp_trigger_distance_ft = 3000.0
        self.phase = XCPhase.GROUND_ROLL
        self.phase_start_time = 0.0
        self.has_turned_to_cruise = False

        self.kp_pitch = 0.8
        self.kd_pitch = 0.6
        self.kp_roll = 1.2
        self.kd_roll = 0.4
        self.turn_bank_angle = np.deg2rad(25.0)

        self.pitch_rotate = np.deg2rad(10.0)
        self.pitch_climb = np.deg2rad(8.0)
        self.pitch_cruise = np.deg2rad(0.0)
        self.pitch_descent = np.deg2rad(-3.0)
        self.pitch_approach = np.deg2rad(-2.0)
        self.pitch_flare = np.deg2rad(5.0)

        self.glideslope_deg = 3.5

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _heading_control(self, target, current, roll, roll_rate):
        error = self._normalize_angle(target - current)
        target_roll = np.clip(error * 0.5, -self.turn_bank_angle, self.turn_bank_angle)
        roll_error = target_roll - roll
        return np.clip(self.kp_roll * roll_error - self.kd_roll * roll_rate, -1.0, 1.0)

    def _pitch_control(self, target, current, pitch_rate):
        error = target - current
        return -np.clip(self.kp_pitch * error - self.kd_pitch * pitch_rate, -1.0, 1.0)

    def compute_action(self, state, sim_time):
        x_ft = state[StateIndex.X] * M_TO_FT
        y_ft = state[StateIndex.Y] * M_TO_FT
        z_ft = state[StateIndex.Z] * M_TO_FT
        altitude_ft = -z_ft

        u_fps = state[StateIndex.U] * M_TO_FT
        v_fps = state[StateIndex.V] * M_TO_FT
        w_fps = state[StateIndex.W] * M_TO_FT
        airspeed_fps = np.sqrt(u_fps**2 + v_fps**2 + w_fps**2)
        climb_rate_fps = -w_fps

        phi = state[StateIndex.PHI]
        theta = state[StateIndex.THETA]
        psi = state[StateIndex.PSI]
        p = state[StateIndex.P]
        q = state[StateIndex.Q]

        dist_to_tp_ft = np.sqrt((x_ft - self.tp_x_ft)**2 + (y_ft - self.tp_y_ft)**2)
        dist_to_threshold_ft = np.sqrt((x_ft - self.threshold_x_ft)**2 +
                                        (y_ft - self.threshold_y_ft)**2)
        dist_to_aimpoint_ft = np.sqrt((x_ft - self.aimpoint_x_ft)**2 +
                                       (y_ft - self.aimpoint_y_ft)**2)

        # Phase transitions
        if self.phase == XCPhase.GROUND_ROLL and airspeed_fps >= self.v_rotate:
            self.phase = XCPhase.ROTATION
            self.phase_start_time = sim_time
        elif self.phase == XCPhase.ROTATION and altitude_ft > self.initial_climb_alt_ft:
            self.phase = XCPhase.INITIAL_CLIMB
        elif self.phase == XCPhase.INITIAL_CLIMB and altitude_ft > self.turn_to_cruise_altitude_ft:
            self.phase = XCPhase.CLIMB
        elif self.phase == XCPhase.CLIMB and altitude_ft >= self.cruise_altitude_ft - 50:
            self.phase = XCPhase.CRUISE_TO_TP
        elif self.phase == XCPhase.CRUISE_TO_TP and dist_to_tp_ft < self.tp_trigger_distance_ft:
            self.phase = XCPhase.TURN_TO_INTERCEPT
        elif self.phase == XCPhase.TURN_TO_INTERCEPT:
            heading_error = abs(self._normalize_angle(psi - self.runway_heading))
            if heading_error < np.deg2rad(10):
                self.phase = XCPhase.INTERCEPT_LEG
        elif self.phase == XCPhase.INTERCEPT_LEG and dist_to_threshold_ft <= 3 * NM_TO_FT:
            self.phase = XCPhase.FINAL_APPROACH
        elif self.phase == XCPhase.FINAL_APPROACH and dist_to_threshold_ft <= self.short_final_distance_ft:
            self.phase = XCPhase.SHORT_FINAL
        elif self.phase == XCPhase.SHORT_FINAL and altitude_ft <= self.flare_altitude_ft:
            self.phase = XCPhase.LANDING
        elif self.phase == XCPhase.LANDING and altitude_ft <= self.touchdown_altitude_ft:
            self.phase = XCPhase.LANDED

        # Control logic by phase
        if self.phase == XCPhase.GROUND_ROLL:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.ROTATION:
            return np.array([1.0, 0.0,
                           self._pitch_control(self.pitch_rotate, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.INITIAL_CLIMB:
            return np.array([1.0,
                           self._heading_control(self.departure_heading, psi, phi, p),
                           self._pitch_control(self.pitch_climb, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.CLIMB:
            if altitude_ft > self.turn_to_cruise_altitude_ft:
                self.has_turned_to_cruise = True
            hdg = self.cruise_heading if self.has_turned_to_cruise else self.departure_heading
            return np.array([1.0,
                           self._heading_control(hdg, psi, phi, p),
                           self._pitch_control(self.pitch_climb, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.CRUISE_TO_TP:
            bearing = np.arctan2(self.tp_y_ft - y_ft, self.tp_x_ft - x_ft)
            alt_error_ft = altitude_ft - self.cruise_altitude_ft
            target_vs_fps = -0.05 * alt_error_ft
            target_vs_fps = np.clip(target_vs_fps, -800/60, 800/60)
            vs_error_fps = climb_rate_fps - target_vs_fps
            pitch_correction = np.deg2rad(-0.6 * vs_error_fps)
            pitch_correction = np.clip(pitch_correction, np.deg2rad(-10.0), np.deg2rad(10.0))
            pitch_target = self.pitch_cruise + pitch_correction
            base_throttle = 0.50
            throttle_correction = -0.001 * alt_error_ft
            throttle = np.clip(base_throttle + throttle_correction, 0.3, 0.7)
            return np.array([throttle,
                           self._heading_control(bearing, psi, phi, p),
                           self._pitch_control(pitch_target, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.TURN_TO_INTERCEPT:
            return np.array([0.7,
                           self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(self.pitch_cruise, theta, q),
                           0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.INTERCEPT_LEG:
            dx_ft = x_ft - self.aimpoint_x_ft
            dy_ft = y_ft - self.aimpoint_y_ft
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            cross_track_ft = dx_ft * (-rwy_dy) + dy_ft * rwy_dx
            intercept_angle = np.clip(-cross_track_ft * 0.0005, -np.deg2rad(20), np.deg2rad(20))
            target_heading = self.runway_heading + intercept_angle
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft
            min_speed_fps = 50.0 * KTS_TO_FPS
            speed_low = airspeed_fps < min_speed_fps

            if speed_low:
                target_descent_fps = -300 / 60
                throttle = 0.9
                spoiler = 0.0
                pitch_base = np.deg2rad(0.0)
            elif alt_error_ft > 2000:
                target_descent_fps = -3000 / 60
                throttle = 0.3
                spoiler = 0.8
                pitch_base = np.deg2rad(-8.0)
            elif alt_error_ft > 1000:
                target_descent_fps = -2500 / 60
                throttle = 0.35
                spoiler = 0.6
                pitch_base = np.deg2rad(-6.0)
            elif alt_error_ft > 500:
                target_descent_fps = -1500 / 60
                throttle = 0.4
                spoiler = 0.4
                pitch_base = np.deg2rad(-4.0)
            elif alt_error_ft > 200:
                target_descent_fps = -1000 / 60
                throttle = 0.45
                spoiler = 0.2
                pitch_base = np.deg2rad(-3.0)
            else:
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.05 * alt_error_ft
                throttle = 0.5
                spoiler = 0.0
                pitch_base = self.pitch_descent

            target_descent_fps = np.clip(target_descent_fps, -3000/60, 0)
            pitch_adjust = (climb_rate_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust

            return np.array([throttle,
                           self._heading_control(target_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 0.3, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.FINAL_APPROACH:
            dx_ft = x_ft - self.aimpoint_x_ft
            dy_ft = y_ft - self.aimpoint_y_ft
            rwy_dx = np.cos(self.runway_heading)
            rwy_dy = np.sin(self.runway_heading)
            cross_track_ft = dx_ft * (-rwy_dy) + dy_ft * rwy_dx
            intercept_angle = np.clip(-cross_track_ft * 0.001, -np.deg2rad(15), np.deg2rad(15))
            target_heading = self.runway_heading + intercept_angle
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            if alt_error_ft > 500:
                target_descent_fps = -2000 / 60
                throttle = 0.3
                spoiler = 0.5
                pitch_base = np.deg2rad(-5.0)
            elif alt_error_ft > 200:
                target_descent_fps = -1000 / 60
                throttle = 0.4
                spoiler = 0.3
                pitch_base = np.deg2rad(-3.0)
            else:
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.1 * alt_error_ft
                throttle = 0.45
                spoiler = 0.0
                pitch_base = self.pitch_approach

            target_descent_fps = np.clip(target_descent_fps, -2000/60, 0)
            pitch_adjust = (climb_rate_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust

            return np.array([throttle,
                           self._heading_control(target_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 0.5, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.SHORT_FINAL:
            target_alt_ft = dist_to_aimpoint_ft * np.tan(np.deg2rad(self.glideslope_deg))
            alt_error_ft = altitude_ft - target_alt_ft

            if alt_error_ft > 300:
                target_descent_fps = -1500 / 60
                throttle = 0.2
                spoiler = 0.4
                pitch_base = np.deg2rad(-4.0)
            elif alt_error_ft > 100:
                target_descent_fps = -800 / 60
                throttle = 0.3
                spoiler = 0.2
                pitch_base = np.deg2rad(-2.0)
            else:
                gs_descent_fps = -airspeed_fps * np.tan(np.deg2rad(self.glideslope_deg))
                target_descent_fps = gs_descent_fps - 0.1 * alt_error_ft
                throttle = 0.35
                spoiler = 0.0
                pitch_base = self.pitch_approach

            target_descent_fps = np.clip(target_descent_fps, -1500/60, 0)
            pitch_adjust = (climb_rate_fps - target_descent_fps) * 0.003
            pitch = pitch_base + pitch_adjust

            return np.array([throttle,
                           self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 0.7, spoiler, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.LANDING:
            pitch_adjust = -climb_rate_fps * 0.005
            pitch = self.pitch_flare + pitch_adjust
            return np.array([0.0,
                           self._heading_control(self.runway_heading, psi, phi, p),
                           self._pitch_control(pitch, theta, q),
                           0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        elif self.phase == XCPhase.LANDED:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
