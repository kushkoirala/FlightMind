# FlightMind - Portable Flight Data Generator

Self-contained package for generating synthetic aviation training data using
AIDA's 6-DOF flight dynamics simulator. Runs on any machine with Python 3.9+
and NumPy — no GPU, no CUDA, no special dependencies.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Test with 10 flights
python generate_flights.py --num-flights 10 --output ./output

# Full run
python generate_flights.py --num-flights 15000 --machine-id 1 --output ./output
```

## Distributed Execution

Each machine uses `--machine-id` to generate unique seed ranges, so output
files never overlap and can be merged directly.

| Machine | ID | Workers | Flights | Command |
|---------|------|---------|---------|---------|
| Dell 7920 | 0 | 38 | 20,000 | `python generate_flights.py --num-flights 20000 --machine-id 0 --num-workers 38 --output ./output` |
| Mac M3 Pro | 1 | 8 | 15,000 | `python generate_flights.py --num-flights 15000 --machine-id 1 --num-workers 8 --output ./output` |
| ROG Ally X | 2 | 10 | 15,000 | `python generate_flights.py --num-flights 15000 --machine-id 2 --num-workers 10 --output ./output` |

Total: **50,000 flights** producing **~250M estimated tokens** across 5 narrative styles.

## Output Structure

```
output/
├── flight_report/     # NTSB-style reports for normal flights
├── instructional/     # Textbook cross-country planning text
├── pilot_log/         # First-person pilot logbook entries
├── atc_comms/         # ATC radio communication transcripts
├── parameter_log/     # Tabular flight data logs
└── manifest_m{ID}.json
```

Files are named `m{machine_id}_flight_{id}_{origin}_{dest}.txt` to avoid
collisions when merging output from multiple machines.

## Merging Output

After all machines finish, copy the output directories together:

```bash
# On the Dell (or any central machine)
# Copy Mac output
cp -r /path/to/mac/output/flight_report/* ./merged/flight_report/
cp -r /path/to/mac/output/instructional/* ./merged/instructional/
# ... repeat for each style and each machine
```

Or use rsync/scp to transfer from remote machines.

## How It Works

1. **Scenario Generation** — Samples random airport pairs, cruise altitudes,
   wind conditions, and weather from 11 Kansas-region airports
2. **Flight Simulation** — Runs CPU-only 6-DOF physics (RK4 integration at
   50 Hz) with a cross-country controller (takeoff → cruise → triangle
   intercept approach → landing)
3. **Narrative Generation** — Converts telemetry into 5 text styles with
   randomized phrasing for diversity

## Performance Estimates

| Machine | CPU | ~Flights/sec | 15K flights |
|---------|-----|-------------|-------------|
| Dell 7920 | 2x Xeon 5118 (48T) | 0.3 | ~18 hours |
| Mac M3 Pro | 12-core ARM | 0.1 | ~42 hours |
| ROG Ally X | Ryzen Z2 Extreme | 0.1 | ~42 hours |

## Airports (11 real Kansas-region airports)

| ICAO | Name | Elevation |
|------|------|-----------|
| SN65 | Lake Waltanna Airport | 1,448 ft |
| KHUT | Hutchinson Regional | 1,542 ft |
| KICT | Eisenhower National (Wichita) | 1,333 ft |
| KAAO | Col. James Jabara | 1,421 ft |
| KEWK | Newton City-County | 1,533 ft |
| K3AU | Augusta Municipal | 1,328 ft |
| KEQA | Capt Jack Thomas El Dorado | 1,378 ft |
| KMCI | Kansas City International | 1,026 ft |
| KWLD | Strother Field (Winfield) | 1,160 ft |
| KPTT | Pratt Regional | 1,953 ft |
| KCNU | Chanute Martin Johnson | 1,002 ft |

## Files

| File | Description |
|------|-------------|
| `generate_flights.py` | Main entry point — scenario gen, flight sim, narrative output |
| `flight_dynamics.py` | 6-DOF flight simulator (CPU/NumPy only, stripped of GPU code) |
| `xc_controller.py` | Cross-country flight controller with triangle intercept pattern |
| `requirements.txt` | Just numpy |
