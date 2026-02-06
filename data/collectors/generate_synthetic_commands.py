"""
Generate Synthetic Flight Command Training Data
=================================================

Creates diverse pilot command → structured action pairs for FlightMind
fine-tuning. Unlike the intent observation converter (which transforms
real AIDA data), this script generates brand-new training examples from
scratch using template expansion and controlled randomization.

WHY synthetic data?
-------------------
Real AIDA observations are dominated by heading/altitude commands during
normal flight phases. But FlightMind also needs to handle:
- Edge cases (extreme headings, unusual altitudes)
- Conversational queries ("what airport is closest?")
- Weather interpretation ("what does this METAR mean?")
- ATC-style phraseology ("Cessna 4523 turn left heading 180")
- Compound commands ("climb to 5000 and turn heading 090")
- Emergency commands ("engine failure, find nearest airport")

Synthetic generation lets us cover the long tail of commands that
rarely appear in normal sim runs but are critical for robustness.

The generated data complements (not replaces) the real AIDA data.
Together they form a balanced fine-tuning dataset.
"""

import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Airport and Navigation Data (Kansas region, matching AIDA)
# ---------------------------------------------------------------------------

AIRPORTS = {
    "SN65": {"name": "Stearman Field", "city": "Benton", "elev": 1404, "runways": ["17/35"]},
    "KHUT": {"name": "Hutchinson Municipal", "city": "Hutchinson", "elev": 1524, "runways": ["13/31", "4/22"]},
    "KAAO": {"name": "Colonel James Jabara", "city": "Wichita", "elev": 1421, "runways": ["18/36"]},
    "KICT": {"name": "Wichita Eisenhower", "city": "Wichita", "elev": 1333, "runways": ["1L/19R", "1R/19L", "14/32"]},
    "KEWK": {"name": "Newton City-County", "city": "Newton", "elev": 1533, "runways": ["17/35"]},
    "KSLN": {"name": "Salina Regional", "city": "Salina", "elev": 1288, "runways": ["17/35", "12/30"]},
    "KMCI": {"name": "Kansas City International", "city": "Kansas City", "elev": 1026, "runways": ["1/19", "9/27"]},
    "KTOP": {"name": "Philip Billard", "city": "Topeka", "elev": 881, "runways": ["13/31"]},
}

# Cessna 172 flight parameters (AIDA primary aircraft)
CESSNA_172 = {
    "Vs0": 48,     # Stall speed flaps down
    "Vs1": 52,     # Stall speed clean
    "Vr": 55,      # Rotation speed
    "Vy": 76,      # Best rate of climb
    "Vx": 62,      # Best angle of climb
    "Va": 99,      # Maneuvering speed
    "Vfe": 85,     # Max flap extended
    "Vno": 129,    # Max structural cruising
    "Vne": 163,    # Never exceed
    "cruise_alt_min": 3000,
    "cruise_alt_max": 10000,
    "service_ceiling": 14200,
}

SYSTEM_PROMPT = "You are FlightMind, an aviation AI copilot for the AIDA autonomous flight system. Parse pilot commands into structured actions and acknowledge clearly."

# ---------------------------------------------------------------------------
# Phases and typical telemetry ranges
# ---------------------------------------------------------------------------

PHASE_PROFILES = {
    "ground":        {"alt": (0, 0),        "spd": (0, 30),    "hdg": (0, 360), "vs": (0, 0)},
    "takeoff":       {"alt": (0, 200),      "spd": (45, 75),   "hdg": (0, 360), "vs": (300, 1000)},
    "initial_climb": {"alt": (200, 1000),   "spd": (70, 90),   "hdg": (0, 360), "vs": (500, 1000)},
    "climb":         {"alt": (1000, 5000),  "spd": (75, 100),  "hdg": (0, 360), "vs": (300, 800)},
    "cruise":        {"alt": (3000, 8000),  "spd": (90, 130),  "hdg": (0, 360), "vs": (-50, 50)},
    "approach":      {"alt": (1500, 4000),  "spd": (70, 110),  "hdg": (0, 360), "vs": (-800, -200)},
    "final":         {"alt": (200, 1500),   "spd": (60, 85),   "hdg": (0, 360), "vs": (-700, -300)},
}


def random_telemetry(phase: str) -> dict:
    """Generate realistic telemetry for a given phase."""
    profile = PHASE_PROFILES.get(phase, PHASE_PROFILES["cruise"])
    return {
        "altitude": random.uniform(*profile["alt"]),
        "airspeed": random.uniform(*profile["spd"]),
        "heading": random.uniform(*profile["hdg"]),
        "vertical_rate": random.uniform(*profile["vs"]),
    }


def format_context(phase: str, telem: dict) -> str:
    parts = [f"PHASE: {phase}"]
    parts.append(f"ALT: {telem['altitude']:.0f}ft")
    parts.append(f"SPD: {telem['airspeed']:.0f}kt")
    parts.append(f"HDG: {telem['heading']:.0f}°")
    if abs(telem.get("vertical_rate", 0)) > 50:
        parts.append(f"VS: {telem['vertical_rate']:+.0f}fpm")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Example Generators
# ---------------------------------------------------------------------------

def gen_heading_commands(n: int) -> list[dict]:
    """Generate heading change commands with diverse phrasings."""
    templates = [
        ("turn to heading {h}", "Roger, turning to heading {h}."),
        ("fly heading {h}", "Copy, heading {h}."),
        ("heading {h}", "Heading {h}, wilco."),
        ("come to heading {h}", "Roger, coming to heading {h}."),
        ("new heading {h}", "New heading {h}."),
        ("turn left heading {h}", "Turning left to heading {h}."),
        ("turn right heading {h}", "Turning right to heading {h}."),
        ("give me heading {h}", "Roger, heading {h}."),
        ("I want heading {h}", "Heading {h}, understood."),
        ("steer {h}", "Steering {h}."),
        ("make heading {h}", "Heading {h}."),
        ("course {h}", "Roger, course {h}."),
        ("go heading {h}", "Heading {h}, wilco."),
        ("bring us to heading {h}", "Coming to heading {h}."),
        ("set course {h} degrees", "Setting course {h} degrees."),
    ]
    examples = []
    for _ in range(n):
        h = random.randint(1, 36) * 10  # Round headings: 10, 20, ..., 360
        if random.random() < 0.3:
            h = random.randint(0, 359)   # Sometimes exact headings
        h = h % 360 or 360

        phase = random.choice(["climb", "cruise", "approach", "initial_climb"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        cmd_template, ack_template = random.choice(templates)
        cmd = cmd_template.format(h=h)
        ack = ack_template.format(h=h)
        structured = json.dumps({"action": "heading", "value": h})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{cmd}",
            "assistant": f"{structured}\n{ack}",
        })
    return examples


def gen_altitude_commands(n: int) -> list[dict]:
    """Generate altitude change commands."""
    templates = [
        ("climb to {a} feet", "Roger, climbing to {a} feet."),
        ("descend to {a} feet", "Descending to {a}."),
        ("altitude {a}", "Altitude {a}, wilco."),
        ("climb and maintain {a}", "Climbing to {a}."),
        ("descend and maintain {a}", "Roger, descending to {a}."),
        ("go to {a} feet", "Going to {a} feet."),
        ("maintain {a}", "Maintaining {a}."),
        ("level off at {a}", "Leveling at {a}."),
        ("take us to {a} feet", "Roger, going to {a}."),
        ("I want {a} feet", "Setting {a} feet."),
        ("new altitude {a}", "New altitude {a}."),
        ("fly at {a}", "Roger, {a} feet."),
        ("bring us up to {a}", "Climbing to {a}."),
        ("bring us down to {a}", "Descending to {a}."),
        ("set altitude {a} feet", "Altitude set to {a}."),
    ]
    common_alts = [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                   5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000]
    examples = []
    for _ in range(n):
        if random.random() < 0.7:
            a = random.choice(common_alts)
        else:
            a = random.randint(10, 120) * 100  # 1000 to 12000 in 100s

        phase = random.choice(["climb", "cruise", "approach"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        cmd_template, ack_template = random.choice(templates)
        cmd = cmd_template.format(a=a)
        ack = ack_template.format(a=a)
        structured = json.dumps({"action": "altitude", "value": a})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{cmd}",
            "assistant": f"{structured}\n{ack}",
        })
    return examples


def gen_speed_commands(n: int) -> list[dict]:
    """Generate speed change commands."""
    templates = [
        ("set speed {s} knots", "Roger, {s} knots."),
        ("speed {s}", "Speed {s}."),
        ("maintain {s} knots", "Maintaining {s} knots."),
        ("slow to {s}", "Slowing to {s} knots."),
        ("accelerate to {s}", "Accelerating to {s} knots."),
        ("fly at {s} knots", "Roger, {s} knots."),
        ("reduce speed to {s}", "Reducing to {s} knots."),
        ("increase speed to {s}", "Increasing to {s} knots."),
        ("{s} knots please", "{s} knots, understood."),
        ("target speed {s}", "Target speed {s} knots."),
    ]
    examples = []
    for _ in range(n):
        s = random.randint(60, 140)
        if random.random() < 0.5:
            s = random.choice([65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 130])

        phase = random.choice(["climb", "cruise", "approach", "final"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        cmd_template, ack_template = random.choice(templates)
        cmd = cmd_template.format(s=s)
        ack = ack_template.format(s=s)
        structured = json.dumps({"action": "speed", "value": s})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{cmd}",
            "assistant": f"{structured}\n{ack}",
        })
    return examples


def gen_landing_commands(n: int) -> list[dict]:
    """Generate landing and diversion commands."""
    templates = [
        ("land at {apt}", "Roger, setting up for {apt}."),
        ("divert to {apt}", "Copy, diverting to {apt}."),
        ("take us to {apt}", "Heading to {apt}."),
        ("go to {apt} and land", "Roger, proceeding to {apt} for landing."),
        ("we're landing at {apt}", "Copy, landing at {apt}."),
        ("proceed to {apt}", "Proceeding to {apt}."),
        ("approach {apt}", "Setting up approach for {apt}."),
        ("I want to land at {apt}", "Roger, landing at {apt}."),
        ("{apt} please", "Roger, heading to {apt}."),
        ("let's go to {apt}", "Copy, going to {apt}."),
        ("nearest airport", "Nearest airport is {apt}. Heading there now."),
        ("find me an airport", "Nearest suitable airport is {apt}."),
    ]
    examples = []
    for _ in range(n):
        apt = random.choice(list(AIRPORTS.keys()))
        info = AIRPORTS[apt]

        phase = random.choice(["cruise", "approach", "climb"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        cmd_template, ack_template = random.choice(templates)
        cmd = cmd_template.format(apt=apt)
        ack = ack_template.format(apt=apt)

        # For airport-specific commands, include name sometimes
        if random.random() < 0.3:
            cmd = cmd_template.format(apt=f"{info['name']}")
            ack = ack_template.format(apt=apt)

        structured = json.dumps({"action": "land", "target": apt})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{cmd}",
            "assistant": f"{structured}\n{ack}",
        })
    return examples


def gen_status_queries(n: int) -> list[dict]:
    """Generate flight status query examples."""
    templates = [
        "what's our altitude?",
        "current altitude?",
        "how high are we?",
        "give me our status",
        "status report",
        "what's our heading?",
        "current heading?",
        "where are we heading?",
        "what's our speed?",
        "how fast are we going?",
        "current airspeed?",
        "where are we?",
        "position report",
        "flight status",
        "report altitude and heading",
        "give me the numbers",
        "what phase are we in?",
        "how are we doing?",
        "report current conditions",
        "what's our vertical rate?",
        "are we climbing or descending?",
    ]
    examples = []
    for _ in range(n):
        phase = random.choice(list(PHASE_PROFILES.keys()))
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)
        query = random.choice(templates)

        phase_desc = {
            "ground": "on the ground",
            "takeoff": "in takeoff",
            "initial_climb": "in initial climb",
            "climb": "climbing",
            "cruise": "in cruise",
            "approach": "on approach",
            "final": "on final approach",
        }.get(phase, phase)

        status = f"Currently {phase_desc}. "
        status += f"Altitude {telem['altitude']:.0f} feet, "
        status += f"airspeed {telem['airspeed']:.0f} knots, "
        status += f"heading {telem['heading']:.0f} degrees."
        if abs(telem.get("vertical_rate", 0)) > 100:
            direction = "climbing" if telem["vertical_rate"] > 0 else "descending"
            status += f" {direction.capitalize()} at {abs(telem['vertical_rate']):.0f} feet per minute."

        structured = json.dumps({"action": "status"})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{query}",
            "assistant": f"{structured}\n{status}",
        })
    return examples


def gen_compound_commands(n: int) -> list[dict]:
    """Generate compound commands (altitude + heading, etc.).

    These are important because pilots often give multi-part commands:
    'Climb to 5000 and turn heading 090'
    """
    templates = [
        ("climb to {a} and turn heading {h}",
         "Roger, climbing to {a} and turning heading {h}."),
        ("heading {h} and climb to {a}",
         "Copy, heading {h}, climbing to {a}."),
        ("descend to {a} and turn heading {h}",
         "Descending to {a}, heading {h}."),
        ("turn to {h} and maintain {a} feet",
         "Heading {h}, maintaining {a} feet."),
        ("fly heading {h} at {a} feet",
         "Heading {h} at {a} feet, wilco."),
        ("{a} feet heading {h}",
         "Roger, {a} feet heading {h}."),
    ]
    examples = []
    for _ in range(n):
        h = random.choice([i * 10 for i in range(1, 37)])
        a = random.choice([2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000])

        phase = random.choice(["climb", "cruise", "approach"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        cmd_template, ack_template = random.choice(templates)
        cmd = cmd_template.format(h=h, a=a)
        ack = ack_template.format(h=h, a=a)
        structured = json.dumps({"action": "compound", "heading": h, "altitude": a})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{cmd}",
            "assistant": f"{structured}\n{ack}",
        })
    return examples


def gen_weather_queries(n: int) -> list[dict]:
    """Generate weather/METAR interpretation examples."""
    metar_examples = [
        ("METAR KHUT 051853Z 18012G20KT 10SM FEW050 SCT100 32/18 A2985",
         "Hutchinson reports winds from 180 at 12 knots gusting 20. Visibility 10 miles. Few clouds at 5,000, scattered at 10,000. Temperature 32C, dewpoint 18C. Altimeter 29.85. VFR conditions, gusty winds on landing."),
        ("METAR KICT 051756Z 22015KT 6SM HZ BKN035 OVC060 30/20 A2990",
         "Wichita reports winds from 220 at 15 knots. Visibility 6 miles in haze. Broken ceiling at 3,500, overcast at 6,000. Temperature 30C, dewpoint 20C. Altimeter 29.90. Marginal VFR due to haze."),
        ("METAR SN65 051900Z AUTO 17008KT 10SM CLR 33/17 A2988",
         "Stearman Field automated report. Winds from 170 at 8 knots. Visibility 10 miles. Clear skies. Temperature 33C, dewpoint 17C. Altimeter 29.88. Excellent VFR conditions."),
        ("METAR KHUT 060053Z 00000KT 10SM CLR 22/15 A2995",
         "Hutchinson reports calm winds. Visibility 10 miles. Clear skies. Temperature 22C, dewpoint 15C. Altimeter 29.95. Perfect VFR conditions."),
        ("METAR KICT 051953Z 20018G28KT 3SM TSRA BKN015 OVC030 24/22 A2975",
         "Wichita reports winds 200 at 18 gusting 28. Visibility 3 miles in thunderstorms and rain. Broken ceiling at 1,500, overcast at 3,000. Temperature 24C, dewpoint 22C. Altimeter 29.75. IFR conditions. Thunderstorm activity — avoid the area."),
        ("METAR KSLN 051856Z 15010KT 10SM SCT080 BKN120 28/16 A2992",
         "Salina reports winds from 150 at 10 knots. Visibility 10 miles. Scattered clouds at 8,000, broken at 12,000. Temperature 28C, dewpoint 16C. Altimeter 29.92. Good VFR conditions."),
    ]

    query_templates = [
        "what does this METAR mean: {metar}",
        "interpret this METAR: {metar}",
        "read me this METAR: {metar}",
        "what's the weather? {metar}",
        "decode: {metar}",
        "weather report: {metar}",
    ]

    examples = []
    for _ in range(n):
        metar, interpretation = random.choice(metar_examples)
        query = random.choice(query_templates).format(metar=metar)

        phase = random.choice(["cruise", "approach", "climb"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        structured = json.dumps({"action": "weather"})
        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{query}",
            "assistant": f"{structured}\n{interpretation}",
        })
    return examples


def gen_aircraft_knowledge(n: int) -> list[dict]:
    """Generate aircraft knowledge Q&A pairs (Cessna 172 focused)."""
    qa_pairs = [
        ("what's our stall speed?",
         "Clean stall speed (Vs1) is 52 knots. With full flaps (Vs0) it's 48 knots."),
        ("what's Vy?",
         "Best rate of climb speed (Vy) is 76 knots."),
        ("what's Vx?",
         "Best angle of climb speed (Vx) is 62 knots. Use for obstacle clearance after takeoff."),
        ("what's the maneuvering speed?",
         "Maneuvering speed (Va) is 99 knots at max gross weight. Do not make full or abrupt control inputs above Va."),
        ("what's the max flap speed?",
         "Maximum flap extended speed (Vfe) is 85 knots. Retract flaps before exceeding this."),
        ("what's Vno?",
         "Maximum structural cruising speed (Vno) is 129 knots. Avoid turbulence above this speed."),
        ("what's Vne?",
         "Never exceed speed (Vne) is 163 knots. Exceeding this can cause structural failure."),
        ("what's our service ceiling?",
         "Service ceiling is 14,200 feet for the Cessna 172."),
        ("what's the rotation speed?",
         "Rotation speed (Vr) is approximately 55 knots."),
        ("what rpm for cruise?",
         "Normal cruise power is approximately 2300-2400 RPM, depending on altitude and temperature."),
        ("how much fuel do we have?",
         "Standard fuel capacity is 56 gallons total, 53 usable. At cruise, consumption is approximately 8-10 gallons per hour."),
        ("what's the max crosswind component?",
         "Maximum demonstrated crosswind component is 15 knots."),
    ]
    examples = []
    for _ in range(n):
        q, a = random.choice(qa_pairs)
        phase = random.choice(list(PHASE_PROFILES.keys()))
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)
        structured = json.dumps({"action": "knowledge"})
        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{q}",
            "assistant": f"{structured}\n{a}",
        })
    return examples


def gen_emergency_commands(n: int) -> list[dict]:
    """Generate emergency/abnormal situation commands."""
    scenarios = [
        ("engine failure",
         "EMERGENCY: Engine failure. Nearest airport is {apt}. Pitching for best glide speed 68 knots. Setting up for emergency landing.",
         {"action": "emergency", "type": "engine_failure", "target": "{apt}"}),
        ("mayday mayday mayday",
         "MAYDAY acknowledged. Squawking 7700. Nearest airport is {apt}, {d:.0f} nm. Setting up emergency approach.",
         {"action": "emergency", "type": "mayday", "target": "{apt}"}),
        ("we have smoke in the cockpit",
         "EMERGENCY: Smoke detected. Diverting to nearest airport {apt}. Recommend opening vents and preparing for precautionary landing.",
         {"action": "emergency", "type": "smoke", "target": "{apt}"}),
        ("low fuel emergency",
         "EMERGENCY: Low fuel. Diverting immediately to {apt}, {d:.0f} nm away. Minimum fuel approach.",
         {"action": "emergency", "type": "low_fuel", "target": "{apt}"}),
        ("instrument failure",
         "Instrument failure noted. Switching to partial panel procedures. Nearest VFR airport is {apt}. Recommend diversion.",
         {"action": "emergency", "type": "instrument_failure", "target": "{apt}"}),
    ]
    examples = []
    for _ in range(n):
        cmd_text, response_template, structured_template = random.choice(scenarios)
        apt = random.choice(list(AIRPORTS.keys()))
        d = random.uniform(5, 25)

        phase = random.choice(["cruise", "climb", "approach"])
        telem = random_telemetry(phase)
        ctx = format_context(phase, telem)

        response = response_template.format(apt=apt, d=d)
        structured = json.dumps({k: v.format(apt=apt) if isinstance(v, str) else v
                                  for k, v in structured_template.items()})

        examples.append({
            "system": SYSTEM_PROMPT,
            "user": f"{ctx}\n{cmd_text}",
            "assistant": f"{structured}\n{response}",
        })
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).resolve().parents[1] / "finetune"
    output_dir.mkdir(exist_ok=True)

    print("Generating synthetic flight command training data...")
    print("=" * 60)

    all_examples = []

    # Generate each category — counts chosen to balance the dataset
    generators = [
        ("heading",     gen_heading_commands,   15000),
        ("altitude",    gen_altitude_commands,  12000),
        ("speed",       gen_speed_commands,      5000),
        ("landing",     gen_landing_commands,    5000),
        ("status",      gen_status_queries,      5000),
        ("compound",    gen_compound_commands,   5000),
        ("weather",     gen_weather_queries,     3000),
        ("knowledge",   gen_aircraft_knowledge,  3000),
        ("emergency",   gen_emergency_commands,  2000),
    ]

    for name, gen_fn, count in generators:
        examples = gen_fn(count)
        all_examples.extend(examples)
        print(f"  {name:>12s}: {len(examples):>6,} examples")

    # Shuffle for training
    random.shuffle(all_examples)

    # Write output
    output_jsonl = output_dir / "synthetic_commands.jsonl"
    output_text = output_dir / "synthetic_commands.txt"

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
    print(f"Total synthetic examples: {len(all_examples):,}")
    json_size = output_jsonl.stat().st_size / 1e6
    text_size = output_text.stat().st_size / 1e6
    print(f"\nOutput files:")
    print(f"  {output_jsonl.name}: {json_size:.1f} MB")
    print(f"  {output_text.name}: {text_size:.1f} MB")


if __name__ == "__main__":
    main()
