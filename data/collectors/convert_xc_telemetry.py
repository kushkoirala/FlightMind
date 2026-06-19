"""
Convert AIDA XC Flight Telemetry → FlightMind Training Data
=============================================================

Reads the cross-country flight recordings (NPZ files) from AIDA and
converts them into training sequences for FlightMind.

Each NPZ contains a complete autonomous flight (SN65→KHUT or KHUT→KAAO)
recorded at 50Hz with full state vectors and control inputs.

We generate two types of training data:

1. **Status narration**: Given a flight state, generate a natural language
   description. This teaches FlightMind to communicate flight status.

2. **Phase transition narration**: At each phase change, describe what
   happened and what's coming next. This teaches FlightMind to understand
   the flow of a flight.

NPZ structure:
    observations: (N, 12) - [x, y, z, u, v, w, phi, theta, psi, p, q, r]
    actions: (N, 7) - [throttle, aileron, elevator, rudder, flaps, spoilers, brakes]
    phases: (N,) - integer phase labels
    distances: (N,) - distance to destination in meters
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

random.seed(42)

SYSTEM_PROMPT = "You are FlightMind, an aviation AI copilot for the AIDA autonomous flight system. Parse pilot commands into structured actions and acknowledge clearly."

# Phase integer → name mapping (from AIDA's generalized_xc_controller.py)
PHASE_NAMES = {
    0: "IDLE",
    1: "GROUND_ROLL",
    2: "ROTATION",
    3: "INITIAL_CLIMB",
    4: "CLIMB",
    5: "CRUISE_TO_TP",
    6: "TURN_TO_INTERCEPT",
    7: "INTERCEPT_LEG",
    8: "FINAL_APPROACH",
    9: "SHORT_FINAL",
    10: "FLARE",
    11: "ROLLOUT",
    12: "LANDING",
    13: "LANDED",
    14: "HANGAR",
}

PHASE_DESCRIPTIONS = {
    "IDLE": "Aircraft is idle on the ramp.",
    "GROUND_ROLL": "Accelerating on the runway for takeoff.",
    "ROTATION": "Rotating — pulling back on the yoke to lift off.",
    "INITIAL_CLIMB": "Airborne, climbing through initial altitude.",
    "CLIMB": "Climbing to cruise altitude.",
    "CRUISE_TO_TP": "In cruise flight, heading toward the turning point.",
    "TURN_TO_INTERCEPT": "Turning to intercept the final approach course.",
    "INTERCEPT_LEG": "On the intercept leg, aligning with the runway.",
    "FINAL_APPROACH": "On final approach, descending toward the runway.",
    "SHORT_FINAL": "Short final — close to the runway, preparing to flare.",
    "FLARE": "Flaring for touchdown — reducing descent rate.",
    "ROLLOUT": "On the ground, decelerating after touchdown.",
    "LANDING": "Landing roll, applying brakes.",
    "LANDED": "Aircraft has come to a stop. Flight complete.",
    "HANGAR": "Taxiing to the hangar.",
}


def state_to_readable(obs: np.ndarray, phase_int: int, distance: float) -> dict:
    """Convert raw state vector to human-readable values.

    State indices (NED frame):
        0: x (north, m)      1: y (east, m)       2: z (down, m — negative = up)
        3: u (forward, m/s)  4: v (side, m/s)     5: w (down, m/s)
        6: phi (roll, rad)   7: theta (pitch, rad) 8: psi (heading, rad)
        9: p (roll rate)    10: q (pitch rate)    11: r (yaw rate)
    """
    x, y, z = obs[0], obs[1], obs[2]
    u, v, w = obs[3], obs[4], obs[5]
    phi, theta, psi = obs[6], obs[7], obs[8]

    # Convert units
    alt_ft = -z * 3.28084  # z is down, so negate for altitude
    airspeed_kt = math.sqrt(u**2 + v**2 + w**2) * 1.94384  # m/s to knots
    heading_deg = math.degrees(psi) % 360
    roll_deg = math.degrees(phi)
    pitch_deg = math.degrees(theta)
    vs_fpm = -w * 196.85  # m/s to fpm (negative w = climbing)
    dist_nm = distance / 1852.0  # meters to nautical miles

    phase_name = PHASE_NAMES.get(phase_int, f"UNKNOWN_{phase_int}")

    return {
        "altitude_ft": alt_ft,
        "airspeed_kt": airspeed_kt,
        "heading_deg": heading_deg,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "vs_fpm": vs_fpm,
        "distance_nm": dist_nm,
        "phase": phase_name,
    }


def controls_to_readable(action: np.ndarray) -> dict:
    """Convert raw action vector to human-readable controls."""
    return {
        "throttle": float(action[0]),
        "aileron": float(action[1]),
        "elevator": float(action[2]),
        "rudder": float(action[3]),
        "flaps": float(action[4]),
        "spoilers": float(action[5]),
        "brakes": float(action[6]),
    }


def generate_status_narration(state: dict) -> str:
    """Generate a natural language status from flight state."""
    phase = state["phase"]
    alt = state["altitude_ft"]
    spd = state["airspeed_kt"]
    hdg = state["heading_deg"]
    vs = state["vs_fpm"]
    dist = state["distance_nm"]
    roll = state["roll_deg"]

    phase_simple = {
        "GROUND_ROLL": "on ground roll",
        "ROTATION": "rotating for takeoff",
        "INITIAL_CLIMB": "in initial climb",
        "CLIMB": "climbing",
        "CRUISE_TO_TP": "in cruise",
        "TURN_TO_INTERCEPT": "turning to intercept",
        "INTERCEPT_LEG": "on intercept leg",
        "FINAL_APPROACH": "on final approach",
        "SHORT_FINAL": "on short final",
        "FLARE": "in the flare",
        "ROLLOUT": "on rollout",
        "LANDING": "on landing roll",
        "LANDED": "landed",
    }.get(phase, phase.lower())

    parts = [f"Currently {phase_simple}."]
    parts.append(f"Altitude {alt:.0f} feet,")
    parts.append(f"airspeed {spd:.0f} knots,")
    parts.append(f"heading {hdg:.0f} degrees.")

    if abs(vs) > 100 and alt > 50:
        direction = "climbing" if vs > 0 else "descending"
        parts.append(f"{direction.capitalize()} at {abs(vs):.0f} feet per minute.")

    if abs(roll) > 5:
        direction = "left" if roll < 0 else "right"
        parts.append(f"Banking {abs(roll):.0f} degrees {direction}.")

    if dist > 0.5:
        parts.append(f"Distance to destination: {dist:.1f} nautical miles.")

    return " ".join(parts)


def generate_phase_transition(old_phase: str, new_phase: str, state: dict) -> str:
    """Generate narration for a phase transition."""
    alt = state["altitude_ft"]
    spd = state["airspeed_kt"]

    transitions = {
        ("GROUND_ROLL", "ROTATION"):
            f"Rotation speed reached at {spd:.0f} knots. Pulling back for takeoff.",
        ("ROTATION", "INITIAL_CLIMB"):
            f"Airborne! Climbing through {alt:.0f} feet at {spd:.0f} knots.",
        ("INITIAL_CLIMB", "CLIMB"):
            f"Cleared initial climb obstacles. Continuing climb through {alt:.0f} feet.",
        ("CLIMB", "CRUISE_TO_TP"):
            f"Reached cruise altitude {alt:.0f} feet. Leveling off at {spd:.0f} knots.",
        ("CRUISE_TO_TP", "TURN_TO_INTERCEPT"):
            f"Approaching turning point. Initiating turn to intercept final approach course.",
        ("TURN_TO_INTERCEPT", "INTERCEPT_LEG"):
            f"Established on intercept leg, heading {state['heading_deg']:.0f} degrees.",
        ("INTERCEPT_LEG", "FINAL_APPROACH"):
            f"On final approach at {alt:.0f} feet. Beginning descent to runway.",
        ("FINAL_APPROACH", "SHORT_FINAL"):
            f"Short final at {alt:.0f} feet, {spd:.0f} knots. Runway in sight.",
        ("SHORT_FINAL", "FLARE"):
            f"Flaring at {alt:.0f} feet. Reducing power for touchdown.",
        ("FLARE", "ROLLOUT"):
            f"Touchdown! On the runway at {spd:.0f} knots. Applying brakes.",
        ("ROLLOUT", "LANDING"):
            f"Decelerating through {spd:.0f} knots on landing roll.",
        ("LANDING", "LANDED"):
            f"Flight complete. Stopped on the runway.",
    }

    key = (old_phase, new_phase)
    if key in transitions:
        return transitions[key]

    return f"Phase change: {old_phase} → {new_phase} at {alt:.0f} feet, {spd:.0f} knots."


def process_flight(npz_path: Path) -> list[dict]:
    """Process a single flight recording into training examples."""
    data = np.load(str(npz_path), allow_pickle=True)
    observations = data["observations"]  # (N, 12)
    actions = data["actions"]            # (N, 7)
    phases = data["phases"]              # (N,)
    distances = data["distances"]        # (N,)

    n_frames = len(observations)
    departure = str(data.get("departure", "???"))
    arrival = str(data.get("arrival", "???"))

    examples = []

    # 1. Sample status narrations (every ~500 frames ≈ 10 seconds at 50Hz)
    sample_interval = 500
    for i in range(0, n_frames, sample_interval):
        state = state_to_readable(observations[i], int(phases[i]), float(distances[i]))

        # Skip ground states with zero altitude (not very interesting)
        if state["altitude_ft"] < 10 and state["phase"] in ("IDLE", "LANDED", "HANGAR"):
            continue

        narration = generate_status_narration(state)
        ctx_parts = [f"PHASE: {state['phase'].lower()}"]
        ctx_parts.append(f"ALT: {state['altitude_ft']:.0f}ft")
        ctx_parts.append(f"SPD: {state['airspeed_kt']:.0f}kt")
        ctx_parts.append(f"HDG: {state['heading_deg']:.0f}°")
        if abs(state["vs_fpm"]) > 50:
            ctx_parts.append(f"VS: {state['vs_fpm']:+.0f}fpm")
        ctx = " | ".join(ctx_parts)

        query = random.choice([
            "status report", "what's our status?", "flight status",
            "give me the numbers", "where are we?", "how are we doing?",
            "report", "current conditions?",
        ])

        structured = json.dumps({"action": "status"})
        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{query}",
            "assistant": f"{structured}\n{narration}",
        })

    # 2. Phase transition narrations
    prev_phase = int(phases[0])
    for i in range(1, n_frames):
        curr_phase = int(phases[i])
        if curr_phase != prev_phase:
            state = state_to_readable(observations[i], curr_phase, float(distances[i]))
            old_name = PHASE_NAMES.get(prev_phase, f"UNKNOWN_{prev_phase}")
            new_name = PHASE_NAMES.get(curr_phase, f"UNKNOWN_{curr_phase}")

            narration = generate_phase_transition(old_name, new_name, state)

            ctx_parts = [f"PHASE: {new_name.lower()}"]
            ctx_parts.append(f"ALT: {state['altitude_ft']:.0f}ft")
            ctx_parts.append(f"SPD: {state['airspeed_kt']:.0f}kt")
            ctx_parts.append(f"HDG: {state['heading_deg']:.0f}°")
            ctx = " | ".join(ctx_parts)

            query = random.choice([
                "what just happened?",
                "status update",
                "phase change?",
                "what's going on?",
                "report",
            ])

            structured = json.dumps({"action": "phase_change", "from": old_name, "to": new_name})
            examples.append({
                "system": SYSTEM_PROMPT,
                "user": f"{ctx}\n{query}",
                "assistant": f"{structured}\n{narration}",
            })

            prev_phase = curr_phase

    # 3. Flight summary (start and end)
    if n_frames > 100:
        final_state = state_to_readable(observations[-1], int(phases[-1]), float(distances[-1]))
        total_time = float(data.get("total_time", n_frames / 50.0))
        minutes = total_time / 60.0

        summary = (
            f"Flight from {departure} to {arrival} complete. "
            f"Total flight time: {minutes:.1f} minutes. "
            f"Final phase: {final_state['phase'].lower().replace('_', ' ')}."
        )

        structured = json.dumps({
            "action": "flight_summary",
            "departure": departure,
            "arrival": arrival,
            "duration_min": round(minutes, 1),
        })

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"PHASE: {final_state['phase'].lower()} | ALT: {final_state['altitude_ft']:.0f}ft\nflight summary",
            "assistant": f"{structured}\n{summary}",
        })

    return examples


def main():
    # Find XC demo files — try WSL path
    possible_dirs = [
        Path(r"\\wsl.localhost\Ubuntu-22.04\home\AIDA\data\xc_demos"),
        Path("/home/AIDA/data/xc_demos"),
    ]

    xc_dir = None
    for d in possible_dirs:
        if d.exists():
            xc_dir = d
            break

    if xc_dir is None:
        print("ERROR: Cannot find AIDA xc_demos directory")
        print("Tried:")
        for d in possible_dirs:
            print(f"  {d}")
        sys.exit(1)

    npz_files = sorted(xc_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} XC flight recordings in {xc_dir}")

    output_dir = Path(__file__).resolve().parents[1] / "finetune"
    output_dir.mkdir(exist_ok=True)

    all_examples = []
    for npz_path in npz_files:
        try:
            examples = process_flight(npz_path)
            all_examples.extend(examples)
            print(f"  {npz_path.name}: {len(examples)} examples")
        except Exception as e:
            print(f"  {npz_path.name}: ERROR - {e}")

    random.shuffle(all_examples)

    # Write output
    output_jsonl = output_dir / "xc_telemetry_pairs.jsonl"
    output_text = output_dir / "xc_telemetry_pairs.txt"

    with open(output_jsonl, "w", encoding="utf-8") as f_json, \
         open(output_text, "w", encoding="utf-8") as f_text:
        for ex in all_examples:
            f_json.write(json.dumps(ex, ensure_ascii=False) + "\n")
            text = (
                f"<|system|>{ex['system']}<|end|>\n"
                f"<|user|>{ex['user']}<|end|>\n"
                f"<|assistant|>{ex['assistant']}<|end|>"
            )
            f_text.write(text + "\n\n")

    print(f"\n{'='*60}")
    print(f"Total telemetry examples: {len(all_examples):,}")
    json_size = output_jsonl.stat().st_size / 1e6
    text_size = output_text.stat().st_size / 1e6
    print(f"Output files:")
    print(f"  {output_jsonl.name}: {json_size:.1f} MB")
    print(f"  {output_text.name}: {text_size:.1f} MB")


if __name__ == "__main__":
    main()
