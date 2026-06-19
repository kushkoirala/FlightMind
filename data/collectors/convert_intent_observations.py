"""
Convert AIDA Intent Observations → FlightMind Fine-tuning Data
================================================================

Reads the 148K+ intent observations logged by AIDA's Bayesian intent system
and converts them into instruction-response pairs for FlightMind fine-tuning.

Each AIDA observation looks like:
    {
        "phase": "cruise",
        "telemetry": {"altitude": 4500, "airspeed": 110, "heading": 270, ...},
        "command": {"action": "heading", "value": 360},
        "validation": {"confidence": 0.92, ...},
        "outcome": "executed"
    }

We convert these into instruction-following format:
    <|system|>You are FlightMind, an aviation copilot.<|end|>
    <|user|>[flight context]\n[natural language command]<|end|>
    <|assistant|>[structured JSON + pilot acknowledgement]<|end|>

The key challenge: AIDA logs structured commands, not the original natural
language. So we *reverse-generate* plausible natural language from the
structured data, with multiple phrasings per command type. This is a form
of data augmentation — for each structured command, we produce several
natural language variants that a pilot might say.

Why this works: The model needs to learn the *mapping* from natural language
to structure. By generating diverse phrasings, we teach it to recognize many
ways of expressing the same intent. This is similar to back-translation in
NMT (Sennrich et al., 2016).
"""

import json
import random
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Reproducibility
random.seed(42)

# ---------------------------------------------------------------------------
# Natural Language Templates
# ---------------------------------------------------------------------------
# For each command action type, we define multiple phrasings a pilot might use.
# {value} gets substituted with the actual number. {value_spoken} gets the
# spoken form (e.g., "two seven zero" for 270).

def spoken_heading(hdg: float) -> str:
    """Convert heading to spoken form: 270 -> 'two seven zero'."""
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    hdg_int = int(round(hdg)) % 360
    hdg_str = f"{hdg_int:03d}"
    return " ".join(digits[d] for d in hdg_str)


def spoken_altitude(alt: float) -> str:
    """Convert altitude to spoken form: 4500 -> 'four thousand five hundred'."""
    alt_int = int(round(alt))
    if alt_int <= 0:
        return "zero"

    thousands = alt_int // 1000
    hundreds = (alt_int % 1000) // 100
    tens = alt_int % 100

    parts = []
    ones_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                  6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                  11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
                  15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
                  19: "nineteen"}
    tens_words = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
                  6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

    if thousands > 0:
        if thousands in ones_words:
            parts.append(f"{ones_words[thousands]} thousand")
        else:
            parts.append(f"{thousands} thousand")

    if hundreds > 0:
        if hundreds in ones_words:
            parts.append(f"{ones_words[hundreds]} hundred")

    if tens > 0:
        if tens in ones_words:
            parts.append(ones_words[tens])
        elif tens % 10 == 0:
            parts.append(tens_words[tens // 10])
        else:
            parts.append(f"{tens_words[tens // 10]} {ones_words[tens % 10]}")

    return " ".join(parts) if parts else str(alt_int)


# Heading command templates
HEADING_TEMPLATES = [
    "turn to heading {value_int}",
    "fly heading {value_int}",
    "heading {value_int}",
    "turn heading {value_int}",
    "come to heading {value_int}",
    "new heading {value_int}",
    "steer heading {value_int}",
    "turn to {value_spoken}",
    "fly heading {value_spoken}",
    "heading {value_spoken}",
    "turn left heading {value_int}",
    "turn right heading {value_int}",
    "make it heading {value_int}",
    "take heading {value_int}",
    "change heading to {value_int}",
    "I want heading {value_int}",
    "proceed heading {value_int}",
    "go to heading {value_int}",
]

# Altitude command templates
ALTITUDE_TEMPLATES = [
    "climb to {value_int} feet",
    "descend to {value_int} feet",
    "altitude {value_int}",
    "climb and maintain {value_int}",
    "descend and maintain {value_int}",
    "go to {value_int} feet",
    "set altitude {value_int}",
    "fly at {value_int} feet",
    "maintain {value_int}",
    "take us to {value_int} feet",
    "change altitude to {value_int}",
    "I want {value_int} feet",
    "new altitude {value_int}",
    "{value_spoken} feet",
    "climb to {value_spoken}",
    "level off at {value_int}",
    "bring us up to {value_int}",
    "go down to {value_int} feet",
]

# Speed command templates
SPEED_TEMPLATES = [
    "set speed {value_int} knots",
    "maintain {value_int} knots",
    "speed {value_int}",
    "fly at {value_int} knots",
    "reduce speed to {value_int}",
    "increase speed to {value_int}",
    "slow to {value_int} knots",
    "accelerate to {value_int}",
    "{value_int} knots please",
    "I want {value_int} knots",
    "target speed {value_int}",
]

# Landing command templates
LAND_TEMPLATES = [
    "land at {target}",
    "divert to {target}",
    "take us to {target}",
    "go to {target} and land",
    "we're landing at {target}",
    "proceed to {target} for landing",
    "set up for landing at {target}",
    "approach {target}",
    "I want to land at {target}",
    "{target} please",
    "let's head to {target}",
    "divert {target}",
]

# Status query templates
STATUS_TEMPLATES = [
    "what's our altitude?",
    "give me our current status",
    "what's our heading?",
    "how fast are we going?",
    "current airspeed?",
    "what's our position?",
    "status report",
    "where are we?",
    "flight status",
    "report altitude and heading",
    "how are we doing?",
    "give me the numbers",
]

# Pilot acknowledgement templates
HEADING_ACKS = [
    "Roger, turning to heading {value_int}.",
    "Copy, heading {value_int}.",
    "Heading {value_int}, wilco.",
    "Roger, {value_spoken}.",
    "Turning to {value_int}.",
    "Coming to heading {value_int}.",
]

ALTITUDE_ACKS = [
    "Roger, climbing to {value_int} feet.",
    "Copy, {value_int} feet.",
    "Descending to {value_int}, wilco.",
    "Roger, maintaining {value_int}.",
    "Going to {value_int} feet.",
]

LAND_ACKS = [
    "Roger, setting up for {target}.",
    "Copy, proceeding to {target}.",
    "Heading to {target} for landing.",
    "Roger, diverting to {target}.",
    "{target}, wilco.",
]


# ---------------------------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------------------------

def format_flight_context(phase: str, telemetry: dict) -> str:
    """Format flight state into a concise context block.

    WHY include context? The model needs to understand the current flight
    state to properly interpret commands. "Climb to 5000" means something
    different during takeoff (we're already climbing) vs. during cruise
    (need to initiate a climb). Phase-aware parsing is critical for safety.
    """
    alt = telemetry.get("altitude", 0)
    spd = telemetry.get("airspeed", 0)
    hdg = telemetry.get("heading", 0)
    vr = telemetry.get("vertical_rate", 0)
    roll = telemetry.get("roll", 0)
    pitch = telemetry.get("pitch", 0)

    parts = [f"PHASE: {phase}"]
    parts.append(f"ALT: {alt:.0f}ft")
    parts.append(f"SPD: {spd:.0f}kt")
    parts.append(f"HDG: {hdg:.0f}°")

    if abs(vr) > 50:
        parts.append(f"VS: {vr:+.0f}fpm")

    return " | ".join(parts)


def format_status_response(phase: str, telemetry: dict) -> str:
    """Generate a natural language status report from telemetry."""
    alt = telemetry.get("altitude", 0)
    spd = telemetry.get("airspeed", 0)
    hdg = telemetry.get("heading", 0)
    vr = telemetry.get("vertical_rate", 0)

    phase_desc = {
        "ground": "on the ground",
        "takeoff": "in takeoff",
        "initial_climb": "in initial climb",
        "climb": "climbing",
        "cruise": "in cruise",
        "approach": "on approach",
        "final": "on final approach",
        "flare": "in the flare",
        "rollout": "on rollout",
        "landing": "landing",
    }.get(phase, phase)

    parts = [f"Currently {phase_desc}."]
    parts.append(f"Altitude {alt:.0f} feet,")
    parts.append(f"airspeed {spd:.0f} knots,")
    parts.append(f"heading {hdg:.0f} degrees.")

    if abs(vr) > 100:
        direction = "climbing" if vr > 0 else "descending"
        parts.append(f"{direction.capitalize()} at {abs(vr):.0f} feet per minute.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main Conversion
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are FlightMind, an aviation AI copilot for the AIDA autonomous flight system. Parse pilot commands into structured actions and acknowledge clearly."


def convert_observation(obs: dict, augment_count: int = 2) -> list[dict]:
    """Convert one AIDA intent observation into fine-tuning examples.

    Each observation produces `augment_count` examples with different
    natural language phrasings. This data augmentation teaches the model
    to handle diverse inputs for the same intent.

    Returns list of {system, user, assistant} dicts.
    """
    phase = obs.get("phase", "unknown")
    telemetry = obs.get("telemetry", {})
    command = obs.get("command", {})
    outcome = obs.get("outcome", "executed")

    action = command.get("action", "unknown")
    value = command.get("value")
    target = command.get("target")

    # Skip invalid or unknown commands
    if action == "unknown" or (value is None and target is None and action not in ("status",)):
        return []

    # Skip low-confidence or rejected commands (noisy labels)
    validation = obs.get("validation", {})
    confidence = validation.get("confidence", 0)
    if outcome == "rejected" or confidence < 0.01:
        return []

    context = format_flight_context(phase, telemetry)
    examples = []

    for _ in range(augment_count):
        nl_command = None
        structured_response = None
        ack = None

        if action == "heading" and value is not None:
            value_int = int(round(value)) % 360
            if value_int == 0:
                value_int = 360
            value_spoken = spoken_heading(value_int)
            template = random.choice(HEADING_TEMPLATES)
            nl_command = template.format(
                value_int=value_int, value_spoken=value_spoken
            )
            structured_response = json.dumps({"action": "heading", "value": value_int})
            ack_template = random.choice(HEADING_ACKS)
            ack = ack_template.format(value_int=value_int, value_spoken=value_spoken)

        elif action == "altitude" and value is not None:
            value_int = int(round(value))
            if value_int < 0:
                value_int = 0
            value_spoken = spoken_altitude(value_int)
            template = random.choice(ALTITUDE_TEMPLATES)
            nl_command = template.format(
                value_int=value_int, value_spoken=value_spoken
            )
            structured_response = json.dumps({"action": "altitude", "value": value_int})
            ack_template = random.choice(ALTITUDE_ACKS)
            ack = ack_template.format(value_int=value_int, value_spoken=value_spoken)

        elif action == "speed" and value is not None:
            value_int = int(round(value))
            template = random.choice(SPEED_TEMPLATES)
            nl_command = template.format(value_int=value_int)
            structured_response = json.dumps({"action": "speed", "value": value_int})
            ack = f"Roger, {value_int} knots."

        elif action == "land" and target:
            template = random.choice(LAND_TEMPLATES)
            nl_command = template.format(target=target.upper())
            structured_response = json.dumps({"action": "land", "target": target.upper()})
            ack_template = random.choice(LAND_ACKS)
            ack = ack_template.format(target=target.upper())

        elif action == "status":
            nl_command = random.choice(STATUS_TEMPLATES)
            status_text = format_status_response(phase, telemetry)
            structured_response = json.dumps({"action": "status"})
            ack = status_text

        if nl_command and structured_response:
            user_msg = f"{context}\n{nl_command}"
            assistant_msg = f"{structured_response}\n{ack}"

            examples.append({
                "system": SYSTEM_PROMPT,
                "user": user_msg,
                "assistant": assistant_msg,
            })

    return examples


def convert_to_text_format(example: dict) -> str:
    """Convert a single example to the text format used for pretraining/fine-tuning.

    Format:
        <|system|>...<|end|>
        <|user|>...<|end|>
        <|assistant|>...<|end|>

    WHY this format? It's the standard instruction-following template used by
    most open-source fine-tuning frameworks (Alpaca, Vicuna, ChatML). The
    special tokens delimit roles, and the model learns to generate only the
    assistant portion. During inference, we provide system + user and let
    the model complete the assistant turn.
    """
    return (
        f"<|system|>{example['system']}<|end|>\n"
        f"<|user|>{example['user']}<|end|>\n"
        f"<|assistant|>{example['assistant']}<|end|>"
    )


def main():
    # Find the intent observations file
    # Try WSL path first, then fallback
    possible_paths = [
        Path(r"\\wsl.localhost\Ubuntu-22.04\home\AIDA\data\intent_observations\intent_observations.jsonl"),
        Path("/home/AIDA/data/intent_observations/intent_observations.jsonl"),
    ]

    input_path = None
    for p in possible_paths:
        if p.exists():
            input_path = p
            break

    if input_path is None:
        print("ERROR: Cannot find intent_observations.jsonl")
        print("Tried:")
        for p in possible_paths:
            print(f"  {p}")
        sys.exit(1)

    output_dir = Path(__file__).resolve().parents[1] / "finetune"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "aida_intent_pairs.jsonl"
    output_text_path = output_dir / "aida_intent_pairs.txt"

    print(f"Reading: {input_path}")
    print(f"Output:  {output_path}")

    # Process observations
    total_obs = 0
    total_examples = 0
    action_counts = {}
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout_json, \
         open(output_text_path, "w", encoding="utf-8") as fout_text:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obs = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            total_obs += 1
            examples = convert_observation(obs, augment_count=2)

            for ex in examples:
                # Write JSONL format (for training scripts)
                fout_json.write(json.dumps(ex, ensure_ascii=False) + "\n")
                # Write text format (for concatenated pretraining)
                fout_text.write(convert_to_text_format(ex) + "\n\n")
                total_examples += 1

                action = json.loads(ex["assistant"].split("\n")[0]).get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1

            if total_obs % 25000 == 0:
                print(f"  Processed {total_obs:,} observations → {total_examples:,} examples...")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Input observations:  {total_obs:,}")
    print(f"Skipped (invalid):   {skipped:,}")
    print(f"Output examples:     {total_examples:,}")
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_examples * 100
        print(f"  {action:>12s}: {count:>8,} ({pct:.1f}%)")

    # File sizes
    json_size = output_path.stat().st_size / 1e6
    text_size = output_text_path.stat().st_size / 1e6
    print(f"\nOutput files:")
    print(f"  {output_path.name}: {json_size:.1f} MB")
    print(f"  {output_text_path.name}: {text_size:.1f} MB")


if __name__ == "__main__":
    main()
