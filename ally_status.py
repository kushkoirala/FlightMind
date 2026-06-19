#!/usr/bin/env python3
"""Lightweight status reporter for ROG Ally - outputs JSON to stdout."""
import json, os, subprocess

def main():
    data = {"flights": 0, "total_files": 0, "cpus": os.cpu_count()}
    fg_root = r"F:\FlightMind\flightgen-portable\output"
    fr_dir = os.path.join(fg_root, "flight_report")
    if os.path.isdir(fr_dir):
        data["flights"] = len([f for f in os.listdir(fr_dir) if f.endswith(".txt")])
    total = 0
    if os.path.isdir(fg_root):
        for d in os.listdir(fg_root):
            dd = os.path.join(fg_root, d)
            if os.path.isdir(dd):
                total += len([f for f in os.listdir(dd) if f.endswith(".txt")])
    data["total_files"] = total
    try:
        r = subprocess.run(["wmic", "cpu", "get", "LoadPercentage", "/value"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("LoadPercentage=") and line.split("=")[1]:
                data["cpu_pct"] = int(line.split("=")[1])
    except Exception:
        pass
    try:
        r = subprocess.run(["wmic", "os", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/value"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("FreePhysicalMemory=") and line.split("=")[1]:
                data["ram_free_kb"] = int(line.split("=")[1])
            elif line.startswith("TotalVisibleMemorySize=") and line.split("=")[1]:
                data["ram_total_kb"] = int(line.split("=")[1])
    except Exception:
        pass
    print(json.dumps(data))

if __name__ == "__main__":
    main()
