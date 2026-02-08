#!/usr/bin/env python3
"""
Portable AIDA Synthetic Aviation Corpus Generator

Generates synthetic aviation training data by running headless flights
and converting telemetry into natural language text in 5 styles.

Designed for distributed execution across multiple machines:
  - Dell 7920:  python generate_flights.py --num-flights 20000 --machine-id 0 --num-workers 24
  - Mac M3 Pro: python generate_flights.py --num-flights 15000 --machine-id 1 --num-workers 8
  - ROG Ally X: python generate_flights.py --num-flights 15000 --machine-id 2 --num-workers 10

Each machine-id produces different seed ranges, so outputs never overlap.

Author: Kushal Koirala (with Claude Code)
Date: February 2026
"""

import numpy as np
import argparse
import sys
import os
import io
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from multiprocessing import Pool
from datetime import datetime

from flight_dynamics import (
    FlightSimulator, StateIndex, ControlIndex, STATE_DIM, CONTROL_DIM
)
from xc_controller import (
    GeneralizedXCController, AirportConfig, XCPhase,
    KANSAS_AIRPORTS, latlon_to_xy,
    M_TO_FT, FT_TO_M, NM_TO_FT, FT_TO_NM, KTS_TO_FPS, FPS_TO_KTS
)

# ============================================================================
# Airport Database (embedded from generate_route_training_data.py)
# ============================================================================

TRAINING_AIRPORTS = {
    "SN65": AirportConfig(
        icao="SN65", name="Lake Waltanna Airport",
        lat=37.594167, lon=-97.615833,
        elevation_ft=1448.0, runway_heading_deg=4.0
    ),
    "KHUT": AirportConfig(
        icao="KHUT", name="Hutchinson Regional Airport",
        lat=38.066167, lon=-97.860500,
        elevation_ft=1542.0, runway_heading_deg=314.0
    ),
    "KICT": AirportConfig(
        icao="KICT", name="Wichita Eisenhower National",
        lat=37.649944, lon=-97.433056,
        elevation_ft=1333.0, runway_heading_deg=14.0
    ),
    "KAAO": AirportConfig(
        icao="KAAO", name="Colonel James Jabara Airport",
        lat=37.747500, lon=-97.221389,
        elevation_ft=1421.0, runway_heading_deg=180.0
    ),
    "KEWK": AirportConfig(
        icao="KEWK", name="Newton City-County Airport",
        lat=38.058333, lon=-97.275556,
        elevation_ft=1533.0, runway_heading_deg=174.0
    ),
    "K3AU": AirportConfig(
        icao="K3AU", name="Augusta Municipal Airport",
        lat=37.670000, lon=-97.077778,
        elevation_ft=1328.0, runway_heading_deg=36.0
    ),
    "KEQA": AirportConfig(
        icao="KEQA", name="Captain Jack Thomas El Dorado Airport",
        lat=37.791389, lon=-96.833889,
        elevation_ft=1378.0, runway_heading_deg=175.0
    ),
    "KMCI": AirportConfig(
        icao="KMCI", name="Kansas City International",
        lat=39.297500, lon=-94.713889,
        elevation_ft=1026.0, runway_heading_deg=19.0
    ),
    "KWLD": AirportConfig(
        icao="KWLD", name="Strother Field",
        lat=37.168611, lon=-97.037528,
        elevation_ft=1160.0, runway_heading_deg=174.0
    ),
    "KPTT": AirportConfig(
        icao="KPTT", name="Pratt Regional Airport",
        lat=37.702530, lon=-98.747000,
        elevation_ft=1953.0, runway_heading_deg=170.0
    ),
    "KCNU": AirportConfig(
        icao="KCNU", name="Chanute Martin Johnson Airport",
        lat=37.667833, lon=-95.486667,
        elevation_ft=1002.0, runway_heading_deg=187.0
    ),
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ScenarioConfig:
    flight_id: int
    origin_icao: str
    destination_icao: str
    cruise_altitude_ft: float
    wind_heading_deg: float
    wind_speed_kts: float
    weather_condition: str
    temperature_c: float
    altimeter_inhg: float
    time_of_day: str
    tail_number: str
    pilot_type: str


@dataclass
class TelemetryPoint:
    time_s: float
    phase: str
    x_ft: float
    y_ft: float
    altitude_ft: float
    airspeed_kts: float
    groundspeed_kts: float
    heading_deg: float
    vs_fpm: float
    bank_deg: float
    pitch_deg: float
    throttle: float
    flaps: float
    dist_to_dest_nm: float


@dataclass
class FlightRecord:
    scenario: ScenarioConfig
    telemetry: List[TelemetryPoint]
    origin_name: str
    origin_elevation_ft: float
    origin_rwy_heading: float
    dest_name: str
    dest_elevation_ft: float
    dest_rwy_heading: float
    total_distance_nm: float
    total_time_min: float
    max_altitude_ft: float
    completed: bool


# ============================================================================
# Scenario Generator
# ============================================================================

TAIL_NUMBERS = [
    "N172SP", "N735QA", "N9517L", "N2346M", "N8421F",
    "N6789P", "N4052B", "N1138K", "N5567R", "N3291C",
    "N7724E", "N4488T", "N9012D", "N6235W", "N1847V",
    "N8876H", "N3164J", "N5590G", "N2713Y", "N4029X",
]

WEATHER_CONDITIONS = [
    ("VFR_clear", 0.50),
    ("VFR_few_clouds", 0.20),
    ("VFR_scattered", 0.15),
    ("VFR_marginal", 0.10),
    ("overcast_high", 0.05),
]


def generate_scenarios(num_flights, seed=42):
    rng = random.Random(seed)
    airport_codes = list(TRAINING_AIRPORTS.keys())
    pairs = [(o, d) for o in airport_codes for d in airport_codes if o != d]

    scenarios = []
    for i in range(num_flights):
        origin, dest = rng.choice(pairs)
        cruise_alt = rng.choice(range(3000, 8500, 500))
        wind_hdg = rng.uniform(0, 360)
        wind_spd = rng.choice([0, 0, 0, 5, 5, 8, 10, 12, 15, 20, 25])
        wx_choices, wx_weights = zip(*WEATHER_CONDITIONS)
        weather = rng.choices(wx_choices, weights=wx_weights, k=1)[0]
        temp = rng.uniform(-5, 35)
        altimeter = round(rng.uniform(29.75, 30.25), 2)

        scenarios.append(ScenarioConfig(
            flight_id=i,
            origin_icao=origin,
            destination_icao=dest,
            cruise_altitude_ft=float(cruise_alt),
            wind_heading_deg=round(wind_hdg, 0),
            wind_speed_kts=float(wind_spd),
            weather_condition=weather,
            temperature_c=round(temp, 1),
            altimeter_inhg=altimeter,
            time_of_day=rng.choice(["morning", "afternoon", "evening"]),
            tail_number=rng.choice(TAIL_NUMBERS),
            pilot_type=rng.choice(["student", "private", "commercial", "ATP"]),
        ))
    return scenarios


# ============================================================================
# Flight Runner
# ============================================================================

def _resolve_airport(icao):
    apt = TRAINING_AIRPORTS[icao]
    if apt.x_ft == 0.0 and apt.y_ft == 0.0 and apt.lat is not None and apt.icao != "SN65":
        ox, oy = latlon_to_xy(apt.lat, apt.lon)
        return AirportConfig(
            icao=apt.icao, name=apt.name, x_ft=ox, y_ft=oy,
            elevation_ft=apt.elevation_ft, runway_heading_deg=apt.runway_heading_deg,
            lat=apt.lat, lon=apt.lon,
        )
    return apt


def run_single_flight(scenario):
    """Run a single flight on CPU. Designed for multiprocessing.Pool."""
    try:
        origin = _resolve_airport(scenario.origin_icao)
        dest = _resolve_airport(scenario.destination_icao)

        dist_ft = np.sqrt((dest.x_ft - origin.x_ft)**2 + (dest.y_ft - origin.y_ft)**2)
        dist_nm = dist_ft * FT_TO_NM
        if dist_nm < 5.0 or dist_nm > 200.0:
            return None

        sim = FlightSimulator(n_instances=1, dt=0.02, use_gpu=False)

        rwy_hdg_rad = np.deg2rad(origin.runway_heading_deg)
        backcourse_rad = rwy_hdg_rad + np.pi
        start_offset_m = 400.0

        initial = np.zeros((1, STATE_DIM), dtype=np.float32)
        initial[0, StateIndex.X] = start_offset_m * np.cos(backcourse_rad)
        initial[0, StateIndex.Y] = start_offset_m * np.sin(backcourse_rad)
        initial[0, StateIndex.Z] = 0.0
        initial[0, StateIndex.U] = 5.0
        initial[0, StateIndex.PSI] = rwy_hdg_rad
        sim.reset(initial)

        origin_config = AirportConfig(
            icao=origin.icao, name=origin.name,
            x_ft=0.0, y_ft=0.0,
            elevation_ft=origin.elevation_ft,
            runway_heading_deg=origin.runway_heading_deg,
            lat=origin.lat, lon=origin.lon,
        )
        dest_config = AirportConfig(
            icao=dest.icao, name=dest.name,
            x_ft=dest.x_ft - origin.x_ft,
            y_ft=dest.y_ft - origin.y_ft,
            elevation_ft=dest.elevation_ft,
            runway_heading_deg=dest.runway_heading_deg,
            lat=dest.lat, lon=dest.lon,
        )

        # Suppress controller printing
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            controller = GeneralizedXCController(
                origin=origin_config, destination=dest_config,
                cruise_altitude_ft=scenario.cruise_altitude_ft,
            )
        finally:
            sys.stdout = old_stdout

        telemetry = []
        sim_time = 0.0
        dt = 0.02
        last_sample = -1.0
        max_time = 7200.0
        step_count = 0
        CTRL_PERIOD = 5
        action = np.zeros(CONTROL_DIM, dtype=np.float32)

        while sim_time < max_time:
            if step_count % CTRL_PERIOD == 0:
                state = sim.get_states()[0]
                action = controller.compute_action(state, sim_time)
                sim.set_controls(action.reshape(1, -1))

                if sim_time - last_sample >= 1.0:
                    last_sample = sim_time
                    x_ft = state[StateIndex.X] * M_TO_FT
                    y_ft = state[StateIndex.Y] * M_TO_FT
                    alt_ft = -state[StateIndex.Z] * M_TO_FT
                    u = state[StateIndex.U]
                    v = state[StateIndex.V]
                    w = state[StateIndex.W]
                    airspeed_kts = np.sqrt(u**2 + v**2 + w**2) * 1.944
                    heading_deg = np.rad2deg(state[StateIndex.PSI]) % 360
                    vs_fpm = -state[StateIndex.W] * M_TO_FT * 60
                    bank_deg = np.rad2deg(state[StateIndex.PHI])
                    pitch_deg = np.rad2deg(state[StateIndex.THETA])

                    ddx = dest_config.x_ft - x_ft
                    ddy = dest_config.y_ft - y_ft
                    dist_to_dest_nm = np.sqrt(ddx**2 + ddy**2) * FT_TO_NM

                    telemetry.append(TelemetryPoint(
                        time_s=round(sim_time, 1), phase=controller.phase.name,
                        x_ft=round(x_ft, 1), y_ft=round(y_ft, 1),
                        altitude_ft=round(alt_ft, 0), airspeed_kts=round(airspeed_kts, 1),
                        groundspeed_kts=round(airspeed_kts, 1), heading_deg=round(heading_deg, 1),
                        vs_fpm=round(vs_fpm, 0), bank_deg=round(bank_deg, 1),
                        pitch_deg=round(pitch_deg, 1),
                        throttle=round(float(action[ControlIndex.THROTTLE]), 2),
                        flaps=round(float(action[ControlIndex.FLAP]), 2),
                        dist_to_dest_nm=round(dist_to_dest_nm, 1),
                    ))

                if controller.phase == XCPhase.LANDED:
                    break

            sim.step()
            sim_time += dt
            step_count += 1

        completed = controller.phase == XCPhase.LANDED
        total_time_min = sim_time / 60.0
        max_alt = max(t.altitude_ft for t in telemetry) if telemetry else 0

        return FlightRecord(
            scenario=scenario, telemetry=telemetry,
            origin_name=origin.name, origin_elevation_ft=origin.elevation_ft,
            origin_rwy_heading=origin.runway_heading_deg,
            dest_name=dest.name, dest_elevation_ft=dest.elevation_ft,
            dest_rwy_heading=dest.runway_heading_deg,
            total_distance_nm=round(dist_nm, 1), total_time_min=round(total_time_min, 1),
            max_altitude_ft=round(max_alt, 0), completed=completed,
        )
    except Exception as e:
        print(f"  [Flight {scenario.flight_id}] Error: {e}", file=sys.stderr)
        return None


# ============================================================================
# Flight Narrator
# ============================================================================

class FlightNarrator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self._months = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]

    def _random_date(self):
        y = self.rng.choice([2023, 2024, 2025])
        m = self.rng.randint(1, 12)
        d = self.rng.randint(1, 28)
        return f"{self._months[m-1]} {d}, {y}"

    def _random_time(self, tod):
        ranges = {"morning": (7, 11), "afternoon": (12, 17), "evening": (17, 20)}
        lo, hi = ranges.get(tod, (8, 17))
        h = self.rng.randint(lo, hi)
        m = self.rng.choice(["00", "15", "30", "45"])
        return f"{h:02d}{m}"

    def _rwy_number(self, heading):
        num = round(heading / 10) % 36
        return str(num if num else 36)

    def _wx_desc(self, wx):
        m = {"VFR_clear": "clear skies", "VFR_few_clouds": "few clouds at 8,000 feet",
             "VFR_scattered": "scattered clouds at 5,500 feet",
             "VFR_marginal": "marginal VFR conditions with scattered clouds at 3,000 feet",
             "overcast_high": "high overcast at 12,000 feet"}
        return m.get(wx, "visual meteorological conditions")

    def _metar(self, s, icao):
        wd = int(s.wind_heading_deg / 10) * 10
        ws = int(s.wind_speed_kts)
        vis = 10 if "clear" in s.weather_condition or "few" in s.weather_condition else self.rng.choice([7, 8, 10])
        t = int(s.temperature_c)
        dp = t - self.rng.randint(3, 10)
        alt = str(s.altimeter_inhg).replace(".", "")
        sky_m = {"VFR_clear": "CLR", "VFR_few_clouds": "FEW080", "VFR_scattered": "SCT055",
                 "VFR_marginal": "SCT030 BKN050", "overcast_high": "OVC120"}
        sky = sky_m.get(s.weather_condition, "CLR")
        return f"{icao} {self._random_time(s.time_of_day)}Z {wd:03d}{ws:02d}KT {vis}SM {sky} {t:02d}/{dp:02d} A{alt}"

    def _phase_start(self, telem, phase):
        for i, t in enumerate(telem):
            if t.phase == phase and (i == 0 or telem[i-1].phase != phase):
                return t
        return None

    def _phase_points(self, telem, phase):
        return [t for t in telem if t.phase == phase]

    def narrate_flight_report(self, r):
        s = r.scenario
        date = self._random_date()
        tl = self._random_time(s.time_of_day)
        rot = self._phase_start(r.telemetry, "ROTATION")
        cruise = self._phase_start(r.telemetry, "CRUISE_TO_TP")
        appr = self._phase_start(r.telemetry, "FINAL_APPROACH")

        t = []
        t.append(f"On {date}, about {tl} central standard time, a Cessna 172S, {s.tail_number}, ")
        t.append(f"departed {r.origin_name} ({s.origin_icao}), for a cross-country flight to ")
        t.append(f"{r.dest_name} ({s.destination_icao}). ")
        t.append(f"The {s.pilot_type} pilot was the sole occupant of the aircraft. ")
        t.append(f"{self._wx_desc(s.weather_condition).capitalize()} prevailed for the departure. ")
        if s.wind_speed_kts > 0:
            t.append(f"Winds were from {int(s.wind_heading_deg):03d} degrees at {int(s.wind_speed_kts)} knots. ")
        t.append(f"The altimeter setting was {s.altimeter_inhg} inches of mercury. ")
        t.append(f"\n\nThe aircraft departed Runway {self._rwy_number(r.origin_rwy_heading)} ")
        if rot:
            t.append(f"and rotated at approximately {rot.airspeed_kts:.0f} knots indicated airspeed. ")
        t.append(f"The pilot climbed to a cruise altitude of {s.cruise_altitude_ft:,.0f} feet MSL")
        if cruise:
            t.append(f", reaching cruise at approximately {cruise.airspeed_kts:.0f} knots. ")
        else:
            t.append(". ")
        t.append(f"The total distance from {s.origin_icao} to {s.destination_icao} was approximately {r.total_distance_nm:.0f} nautical miles. ")
        if appr:
            t.append(f"\n\nThe pilot performed a triangle intercept approach to Runway {self._rwy_number(r.dest_rwy_heading)} at {s.destination_icao}. ")
            t.append(f"The approach was stabilized at approximately {appr.airspeed_kts:.0f} knots on a heading of {r.dest_rwy_heading:.0f} degrees. ")
        status = "landed without incident" if r.completed else "the flight was terminated"
        t.append(f"The aircraft {status}. ")
        t.append(f"Total flight time was approximately {r.total_time_min:.0f} minutes. ")
        t.append(f"The maximum altitude reached during the flight was {r.max_altitude_ft:,.0f} feet MSL.")
        return "".join(t)

    def narrate_instructional(self, r):
        s = r.scenario
        o = TRAINING_AIRPORTS[s.origin_icao]
        d = TRAINING_AIRPORTS[s.destination_icao]
        # Use resolved positions for course calculation
        o_resolved = _resolve_airport(s.origin_icao)
        d_resolved = _resolve_airport(s.destination_icao)
        course = np.rad2deg(np.arctan2(d_resolved.y_ft - o_resolved.y_ft,
                                        d_resolved.x_ft - o_resolved.x_ft)) % 360
        cpts = self._phase_points(r.telemetry, "CRUISE_TO_TP")
        avg_kts = np.mean([p.airspeed_kts for p in cpts]) if cpts else 110.0
        ete = (r.total_distance_nm / (avg_kts / 60)) if avg_kts > 0 else 0

        t = []
        t.append(f"Cross-Country Flight Planning: {s.origin_icao} to {s.destination_icao}\n\n")
        t.append(f"Route: {r.origin_name} ({s.origin_icao}) to {r.dest_name} ({s.destination_icao})\n")
        t.append(f"Distance: {r.total_distance_nm:.0f} nautical miles\n")
        t.append(f"Course: {course:.0f} degrees true\n")
        t.append(f"Cruise Altitude: {s.cruise_altitude_ft:,.0f} feet MSL\n")
        t.append(f"Estimated Cruise Speed: {avg_kts:.0f} KTAS\n")
        t.append(f"Estimated En-Route Time: {ete:.0f} minutes\n\n")
        t.append(f"Departure Procedure:\n")
        t.append(f"Depart Runway {self._rwy_number(r.origin_rwy_heading)} (heading {r.origin_rwy_heading:.0f} degrees). ")
        t.append(f"Rotate at 54 KIAS, climb at Vy (74 KIAS) to {s.cruise_altitude_ft:,.0f} feet MSL. ")
        t.append(f"After reaching 1,000 feet AGL, turn to the cruise heading of {course:.0f} degrees.\n\n")
        t.append(f"En-Route:\n")
        t.append(f"Maintain {s.cruise_altitude_ft:,.0f} feet MSL at cruise power ({avg_kts:.0f} KTAS). ")
        if s.wind_speed_kts > 0:
            t.append(f"Forecast winds are {int(s.wind_heading_deg):03d} at {int(s.wind_speed_kts)} knots. ")
        t.append(f"Monitor fuel burn and cross-check position with GPS or pilotage.\n\n")
        t.append(f"Arrival Procedure:\n")
        t.append(f"The approach to {s.destination_icao} Runway {self._rwy_number(r.dest_rwy_heading)} uses a triangle intercept pattern. ")
        t.append(f"The turning point is positioned 10 NM behind the runway threshold on the backcourse (heading {(r.dest_rwy_heading + 180) % 360:.0f}). ")
        t.append(f"After reaching the turning point, execute a procedure turn to intercept the final approach course of {r.dest_rwy_heading:.0f} degrees. ")
        t.append(f"Descend on a 3.5-degree glideslope, configuring for landing: flaps full, airspeed 65 KIAS. ")
        t.append(f"Cross the threshold at 50 feet AGL, flare, and touch down.\n\n")
        t.append(f"Weather:\n")
        t.append(f"METAR: {self._metar(s, s.origin_icao)}\n")
        t.append(f"{self._wx_desc(s.weather_condition).capitalize()}.\n")
        t.append(f"Temperature: {s.temperature_c:.0f} C, Altimeter: {s.altimeter_inhg} inHg.")
        return "".join(t)

    def narrate_pilot_log(self, r):
        s = r.scenario
        tz = self._random_time(s.time_of_day)
        rot = self._phase_start(r.telemetry, "ROTATION")
        clb = self._phase_start(r.telemetry, "CLIMB")
        crs = self._phase_start(r.telemetry, "CRUISE_TO_TP")
        trn = self._phase_start(r.telemetry, "TURN_TO_INTERCEPT")
        fin = self._phase_start(r.telemetry, "FINAL_APPROACH")

        t = []
        t.append(f"PILOT FLIGHT LOG - {s.tail_number}\n")
        t.append(f"Date: {self._random_date()}\n")
        t.append(f"Route: {s.origin_icao} -> {s.destination_icao}\n")
        t.append(f"Aircraft: Cessna 172S ({s.tail_number})\n\n")
        t.append(f"Departed {s.origin_icao} RWY {self._rwy_number(r.origin_rwy_heading)} at {tz}Z. ")
        if rot:
            t.append(f"Rotated {rot.airspeed_kts:.0f} KIAS. ")
        t.append(f"Positive rate, gear fixed. ")
        if clb:
            t.append(f"Climbing through {clb.altitude_ft:,.0f} ft at {clb.vs_fpm:+.0f} FPM. ")
        if crs:
            t.append(f"Leveled off at {s.cruise_altitude_ft:,.0f} ft, set cruise power - {crs.airspeed_kts:.0f} KTAS. ")
        if s.wind_speed_kts > 0:
            t.append(f"Wind {int(s.wind_heading_deg):03d}/{int(s.wind_speed_kts):02d}. ")
        if trn:
            t.append(f"\nApproaching {s.destination_icao}, initiated triangle intercept at {trn.dist_to_dest_nm:.0f} NM. ")
        if fin:
            t.append(f"On final RWY {self._rwy_number(r.dest_rwy_heading)}, {fin.airspeed_kts:.0f} KIAS, {fin.altitude_ft:,.0f} ft. ")
            t.append(f"Glideslope 3.5 degrees, flaps full. ")
        if r.completed:
            t.append(f"Touchdown. ")
        else:
            t.append(f"Flight terminated (did not complete approach). ")
        t.append(f"\n\nTotal time: {r.total_time_min:.0f} min. ")
        t.append(f"Distance: {r.total_distance_nm:.0f} NM. ")
        t.append(f"Max altitude: {r.max_altitude_ft:,.0f} ft.")
        return "".join(t)

    def narrate_atc(self, r):
        s = r.scenario
        tn = s.tail_number
        sq = self.rng.randint(1200, 7699)
        o_resolved = _resolve_airport(s.origin_icao)
        d_resolved = _resolve_airport(s.destination_icao)
        course = np.rad2deg(np.arctan2(d_resolved.y_ft - o_resolved.y_ft,
                                        d_resolved.x_ft - o_resolved.x_ft))
        cardinal = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
        ci = int(((course % 360) + 22.5) / 45) % 8
        clb = self._phase_start(r.telemetry, "CLIMB")
        crs = self._phase_start(r.telemetry, "CRUISE_TO_TP")

        t = []
        t.append(f"--- ATC Communications Log: {s.origin_icao} to {s.destination_icao} ---\n\n")
        t.append(f'{tn}: "{r.origin_name} traffic, Cessna {tn[-3:]}, ')
        t.append(f"departing Runway {self._rwy_number(r.origin_rwy_heading)}, ")
        t.append(f'departing to the {cardinal[ci]}, {r.origin_name} traffic."\n\n')
        if clb:
            t.append(f'{tn}: "Wichita Approach, Cessna {tn[-3:]}, ')
            t.append(f"{clb.dist_to_dest_nm:.0f} miles {cardinal[(ci+4)%8]} of {s.destination_icao}, ")
            t.append(f"climbing through {clb.altitude_ft:,.0f} for {s.cruise_altitude_ft:,.0f}, ")
            t.append(f'VFR to {r.dest_name}."\n\n')
            t.append(f'Approach: "{tn}, Wichita Approach, squawk {sq}, altimeter {s.altimeter_inhg}."\n\n')
            t.append(f'{tn}: "Squawk {sq}, {s.altimeter_inhg}, {tn[-3:]}."\n\n')
        if crs:
            t.append(f'Approach: "{tn}, radar contact, {crs.dist_to_dest_nm:.0f} miles from {s.destination_icao}, ')
            t.append(f'maintain VFR at {s.cruise_altitude_ft:,.0f}."\n\n')
            t.append(f'{tn}: "Maintain {s.cruise_altitude_ft:,.0f}, {tn[-3:]}."\n\n')
        t.append(f'Approach: "{tn}, {s.destination_icao} weather: ')
        t.append(f"wind {int(s.wind_heading_deg):03d} at {int(s.wind_speed_kts)}, altimeter {s.altimeter_inhg}, ")
        t.append(f'expect Runway {self._rwy_number(r.dest_rwy_heading)} at {s.destination_icao}. Report the field in sight."\n\n')
        t.append(f'{tn}: "Field in sight, {tn[-3:]}."\n\n')
        t.append(f'Approach: "{tn}, frequency change approved, contact {s.destination_icao} traffic advisory."\n\n')
        t.append(f'{tn}: "{r.dest_name} traffic, Cessna {tn[-3:]}, ')
        t.append(f'straight-in Runway {self._rwy_number(r.dest_rwy_heading)}, 3 mile final, {r.dest_name} traffic."\n\n')
        if r.completed:
            t.append(f'{tn}: "{r.dest_name} traffic, Cessna {tn[-3:]}, ')
            t.append(f'clear of Runway {self._rwy_number(r.dest_rwy_heading)}, {r.dest_name} traffic."\n')
        return "".join(t)

    def narrate_parameter_log(self, r):
        s = r.scenario
        t = []
        t.append(f"FLIGHT DATA LOG - {s.tail_number} - {s.origin_icao} to {s.destination_icao}\n")
        t.append(f"Aircraft: Cessna 172S | Date: {self._random_date()}\n")
        t.append(f"Cruise Alt: {s.cruise_altitude_ft:,.0f} ft | Distance: {r.total_distance_nm:.0f} NM\n")
        t.append(f"{'TIME':>8} {'PHASE':<20} {'ALT_FT':>7} {'IAS_KT':>7} {'HDG':>5} {'VS_FPM':>7} {'BANK':>6} {'THR':>5} {'FLP':>5} {'DIST_NM':>8}\n")
        t.append(f"{'-'*8} {'-'*20} {'-'*7} {'-'*7} {'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*8}\n")

        prev_phase = None
        for i, p in enumerate(r.telemetry):
            is_trans = (p.phase != prev_phase)
            is_int = (int(p.time_s) % 5 == 0)
            prev_phase = p.phase
            if is_trans or is_int:
                m, sec = divmod(int(p.time_s), 60)
                ts = f"{m:02d}:{sec:02d}"
                mk = "*" if is_trans else " "
                t.append(f"{ts:>7}{mk} {p.phase:<20} {p.altitude_ft:>7.0f} {p.airspeed_kts:>7.1f} {p.heading_deg:>5.0f} {p.vs_fpm:>+7.0f} {p.bank_deg:>+6.1f} {p.throttle:>5.0%} {p.flaps:>5.0%} {p.dist_to_dest_nm:>8.1f}\n")

        t.append(f"\nFlight {'COMPLETED' if r.completed else 'INCOMPLETE'} | Total time: {r.total_time_min:.1f} min | Max alt: {r.max_altitude_ft:,.0f} ft\n")
        return "".join(t)

    def narrate_all(self, record):
        return {
            "flight_report": self.narrate_flight_report(record),
            "instructional": self.narrate_instructional(record),
            "pilot_log": self.narrate_pilot_log(record),
            "atc_comms": self.narrate_atc(record),
            "parameter_log": self.narrate_parameter_log(record),
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic aviation training corpus from AIDA simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distributed execution (each machine gets unique flights via --machine-id):

  Dell 7920 (24 threads):
    python generate_flights.py --num-flights 20000 --machine-id 0 --num-workers 24 --output ./output

  Mac M3 Pro (8 cores):
    python generate_flights.py --num-flights 15000 --machine-id 1 --num-workers 8 --output ./output

  ROG Ally X (12 threads):
    python generate_flights.py --num-flights 15000 --machine-id 2 --num-workers 10 --output ./output
        """,
    )
    parser.add_argument("--num-flights", type=int, default=100,
                        help="Number of flights to generate (default: 100)")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="CPU worker processes (default: auto-detect)")
    parser.add_argument("--machine-id", type=int, default=0,
                        help="Machine ID for seed offset (0=Dell, 1=Mac, 2=Ally)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base seed (default: 42)")
    parser.add_argument("--styles", type=str, default="all",
                        help="Comma-separated styles or 'all'")
    args = parser.parse_args()

    # Auto-detect worker count
    if args.num_workers is None:
        import os
        cpu_count = os.cpu_count() or 4
        # Leave 2 cores free for system
        args.num_workers = max(1, cpu_count - 2)

    # Compute unique seed from machine-id
    seed = args.seed_base + args.machine_id * 1_000_000

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  AIDA Synthetic Aviation Corpus Generator (Portable)")
    print("=" * 70)
    print(f"Flights:    {args.num_flights}")
    print(f"Output:     {output_dir.resolve()}")
    print(f"Workers:    {args.num_workers} CPU processes")
    print(f"Machine ID: {args.machine_id}")
    print(f"Seed:       {seed}")
    print(f"Airports:   {len(TRAINING_AIRPORTS)}")
    print()

    # Step 1: Generate scenarios
    print("[1/4] Generating flight scenarios...")
    scenarios = generate_scenarios(args.num_flights, seed=seed)
    print(f"  Generated {len(scenarios)} scenarios")

    # Prepare narrator and output directories upfront
    narrator = FlightNarrator(seed=seed)
    styles_to_gen = ["flight_report", "instructional", "pilot_log", "atc_comms", "parameter_log"]
    if args.styles != "all":
        styles_to_gen = [s.strip() for s in args.styles.split(",")]
    for style in styles_to_gen:
        (output_dir / style).mkdir(exist_ok=True)

    # Step 2: Run flights and write narratives incrementally
    print(f"\n[2/4] Running {len(scenarios)} headless flights ({args.num_workers} workers)...")
    print(f"       Narratives written to disk as each flight completes.")
    t0 = time.time()

    total_files = 0
    total_chars = 0
    completed_count = 0
    total_done = 0

    def _write_narratives(record):
        """Write all narrative styles for a single flight record. Returns (files, chars)."""
        files = 0
        chars = 0
        narratives = narrator.narrate_all(record)
        for style, text in narratives.items():
            if style not in styles_to_gen:
                continue
            fid = record.scenario.flight_id
            mid = args.machine_id
            fname = f"m{mid}_flight_{fid:05d}_{record.scenario.origin_icao}_{record.scenario.destination_icao}.txt"
            fpath = output_dir / style / fname
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)
            files += 1
            chars += len(text)
        return files, chars

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for result in pool.imap_unordered(run_single_flight, scenarios, chunksize=4):
                total_done += 1
                if result is not None:
                    if result.completed:
                        completed_count += 1
                    f_count, c_count = _write_narratives(result)
                    total_files += f_count
                    total_chars += c_count
                if total_done % 10 == 0 or total_done == len(scenarios):
                    elapsed = time.time() - t0
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta_min = ((len(scenarios) - total_done) / rate / 60) if rate > 0 else 0
                    print(f"  {total_done}/{len(scenarios)} flights done "
                          f"({completed_count} landed, {rate:.1f} flights/s, "
                          f"ETA: {eta_min:.0f} min)", end="\r")
        print()
    else:
        for i, sc in enumerate(scenarios):
            result = run_single_flight(sc)
            total_done += 1
            if result is not None:
                if result.completed:
                    completed_count += 1
                f_count, c_count = _write_narratives(result)
                total_files += f_count
                total_chars += c_count
            if (i + 1) % 5 == 0:
                print(f"  Flight {i+1}/{len(scenarios)}...", end="\r")
        print()

    elapsed = time.time() - t0
    est_tokens = total_chars // 4
    print(f"  Completed: {completed_count}/{total_done} flights landed ({elapsed:.1f}s, "
          f"{total_done/elapsed:.1f} flights/s)")
    print(f"  {total_files} narrative files written ({total_chars:,} chars, ~{est_tokens:,} tokens)")

    # Step 4: Write manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "machine_id": args.machine_id,
        "seed": seed,
        "num_flights": len(records),
        "completed_flights": completed,
        "styles": styles_to_gen,
        "total_files": total_files,
        "total_characters": total_chars,
        "estimated_tokens": est_tokens,
        "airports_used": list(TRAINING_AIRPORTS.keys()),
        "elapsed_seconds": round(elapsed, 1),
        "flights_per_second": round(len(records) / elapsed, 2) if elapsed > 0 else 0,
    }
    manifest_name = f"manifest_m{args.machine_id}.json"
    with open(output_dir / manifest_name, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[4/4] Manifest written to {output_dir / manifest_name}")

    print(f"\n{'=' * 70}")
    print(f"  DONE! {total_files} narrative files, ~{est_tokens:,} estimated tokens")
    print(f"  Machine {args.machine_id} | Seed {seed} | {elapsed:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
