#!/usr/bin/env python3
"""
FlightMind Training Dashboard
Responsive mobile/desktop monitoring for training and data generation.
Run: python dashboard.py
"""

import http.server
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PORT = 8080
REFRESH_SECONDS = 30

DELL_FLIGHTGEN_DIR = Path(r"D:\FlightMind\data\raw\aida_synthetic")
DELL_FLIGHTGEN_TOTAL = 20000
LORA_TOTAL_STEPS = 2000
LORA_CHECKPOINT_DIR = Path(r"D:\FlightMind\checkpoints")
ALLY_SSH = "AIDA@10.0.1.123"
ALLY_FLIGHTGEN_TOTAL = 15000
CLAUDE_TASKS_DIR = Path(os.path.expanduser(
    r"~\AppData\Local\Temp\claude\c--Users-Administrator-Desktop\tasks"))

# ============================================================================
# LOG AUTO-DETECTION + PARSING
# ============================================================================

_log_cache = {}

def find_latest_log(keyword, secondary=None):
    cache_key = f"{keyword}:{secondary}"
    now = time.time()
    if cache_key in _log_cache and now - _log_cache[cache_key][1] < 120:
        p = _log_cache[cache_key][0]
        if p and p.exists():
            return p
    if not CLAUDE_TASKS_DIR.exists():
        return None
    best, best_mt = None, 0
    for f in CLAUDE_TASKS_DIR.glob("*.output"):
        try:
            mt = f.stat().st_mtime
            if mt < best_mt:
                continue
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                hdr = fh.read(800)
            if keyword not in hdr:
                continue
            if secondary and secondary not in hdr:
                continue
            best, best_mt = f, mt
        except Exception:
            continue
    _log_cache[cache_key] = (best, now)
    return best

def parse_flightgen_log(log_path):
    if not log_path or not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(
            r'(\d+)/(\d+) flights done \((\d+) landed, ([\d.]+) flights/s, ETA: (\d+) min\)', text)
        if matches:
            m = matches[-1]
            return {"done": int(m[0]), "total": int(m[1]), "landed": int(m[2]),
                    "rate": float(m[3]), "eta_min": int(m[4])}
    except Exception:
        pass
    return None

# ============================================================================
# HARDWARE INFO (collected once at startup)
# ============================================================================

def get_hardware_info():
    info = {"hostname": socket.gethostname(), "os": platform.platform(),
            "cpu_model": "", "cpu_cores": os.cpu_count() or 0}
    try:
        r = subprocess.run(["wmic", "cpu", "get", "Name", "/value"],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("Name="):
                info["cpu_model"] = line.split("=", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = [x.strip() for x in r.stdout.strip().split(",")]
            info["gpu_model"] = p[0]
            info["gpu_driver"] = p[1] if len(p) > 1 else ""
    except Exception:
        pass
    try:
        import psutil
        disk = psutil.disk_usage("D:\\")
        info["disk_total_gb"] = round(disk.total / 1e9, 0)
        info["disk_free_gb"] = round(disk.free / 1e9, 0)
    except Exception:
        pass
    return info

# ============================================================================
# DATA COLLECTORS
# ============================================================================

def _safe_int(s):
    try: return int(s)
    except (ValueError, TypeError): return 0

def _safe_float(s):
    try: return float(s)
    except (ValueError, TypeError): return 0.0

def get_dell_system():
    data = {"cpu_pct": 0, "cpu_cores": os.cpu_count() or 0,
            "ram_used_gb": 0, "ram_total_gb": 0, "ram_pct": 0}
    try:
        import psutil
        data["cpu_pct"] = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        data["ram_used_gb"] = round(mem.used / 1e9, 1)
        data["ram_total_gb"] = round(mem.total / 1e9, 1)
        data["ram_pct"] = mem.percent
        disk = psutil.disk_usage("D:\\")
        data["disk_used_gb"] = round(disk.used / 1e9, 1)
        data["disk_free_gb"] = round(disk.free / 1e9, 1)
    except ImportError:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            p = [x.strip() for x in r.stdout.strip().split(",")]
            data.update({"gpu_name": p[0], "gpu_temp": _safe_int(p[1]),
                         "gpu_util": _safe_int(p[2]),
                         "gpu_vram_used": _safe_int(p[3]),
                         "gpu_vram_total": _safe_int(p[4]),
                         "gpu_power": _safe_float(p[5])})
    except Exception:
        pass
    return data

def get_dell_flightgen():
    result = {"completed": 0, "total": DELL_FLIGHTGEN_TOTAL, "pct": 0,
              "total_files": 0, "rate": 0, "eta_min": 0, "landed": 0}
    if DELL_FLIGHTGEN_DIR.exists():
        fr = DELL_FLIGHTGEN_DIR / "flight_report"
        if fr.exists():
            result["completed"] = len(list(fr.glob("*.txt")))
        tf = 0
        for sd in DELL_FLIGHTGEN_DIR.iterdir():
            if sd.is_dir():
                tf += len(list(sd.glob("*.txt")))
        result["total_files"] = tf
    if DELL_FLIGHTGEN_TOTAL > 0:
        result["pct"] = round(100 * result["completed"] / DELL_FLIGHTGEN_TOTAL, 2)
    log = find_latest_log("Synthetic Aviation", "Machine ID: 0")
    prog = parse_flightgen_log(log)
    if prog:
        result.update({"rate": prog["rate"], "eta_min": prog["eta_min"],
                       "landed": prog["landed"]})
    return result

def get_lora_status():
    result = {"active": False, "step": 0, "total_steps": LORA_TOTAL_STEPS,
              "pct": 0, "loss": 0, "lr": 0, "grad_norm": 0, "tok_s": 0,
              "loss_history": [], "checkpoint_step": 0, "best_loss": 999}
    try:
        for c in LORA_CHECKPOINT_DIR.glob("finetune_step_*.pt"):
            m = re.search(r'step_(\d+)', c.name)
            if m:
                s = int(m.group(1))
                if s > result["checkpoint_step"]:
                    result["checkpoint_step"] = s
    except Exception:
        pass
    log_path = find_latest_log("LoRA")
    if not log_path or not log_path.exists():
        return result
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        pat = re.compile(
            r'step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)'
            r'\s+\|\s+grad_norm\s+([\d.]+)\s+\|\s+(\d+)\s+tok/s')
        matches = pat.findall(text)
        if matches:
            best = 999
            for m in matches:
                l = float(m[1])
                if l < best:
                    best = l
                result["loss_history"].append({"step": int(m[0]), "loss": l})
            result["best_loss"] = round(best, 4)
            last = matches[-1]
            result.update({
                "step": int(last[0]), "loss": float(last[1]),
                "lr": float(last[2]), "grad_norm": float(last[3]),
                "tok_s": int(last[4]),
                "pct": round(100 * int(last[0]) / LORA_TOTAL_STEPS, 1)})
            mtime = log_path.stat().st_mtime
            result["active"] = (time.time() - mtime) < 120
            result["loss_history"] = result["loss_history"][-200:]
    except Exception:
        pass
    return result

def get_ally_status():
    result = {"online": False, "cpu_pct": 0, "cpu_cores": 0,
              "ram_used_gb": 0, "ram_total_gb": 0,
              "flights": 0, "total_files": 0,
              "total": ALLY_FLIGHTGEN_TOTAL, "pct": 0,
              "rate": 0, "eta_min": 0}
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes", ALLY_SSH,
             r"C:\Python311\python.exe F:\FlightMind\flightgen-portable\status.py"],
            capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            for line in reversed(r.stdout.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    d = json.loads(line)
                    result["online"] = True
                    result["flights"] = d.get("flights", 0)
                    result["total_files"] = d.get("total_files", 0)
                    result["cpu_cores"] = d.get("cpus", 0)
                    result["cpu_pct"] = d.get("cpu_pct", 0)
                    if "ram_total_kb" in d:
                        result["ram_total_gb"] = round(d["ram_total_kb"] / 1e6, 1)
                    if "ram_free_kb" in d and "ram_total_kb" in d:
                        result["ram_used_gb"] = round(
                            (d["ram_total_kb"] - d["ram_free_kb"]) / 1e6, 1)
                    result["pct"] = round(
                        100 * result["flights"] / ALLY_FLIGHTGEN_TOTAL, 2)
                    break
    except Exception:
        pass
    log = find_latest_log("Synthetic Aviation", "Machine ID: 2")
    prog = parse_flightgen_log(log)
    if prog:
        result.update({"rate": prog["rate"], "eta_min": prog["eta_min"]})
    return result

# ============================================================================
# BACKGROUND COLLECTOR
# ============================================================================

class StatusCollector(threading.Thread):
    daemon = True

    def __init__(self, hardware_info):
        super().__init__()
        self._data = {}
        self._lock = threading.Lock()
        self._history = []
        self._hw = hardware_info
        self._start_time = time.time()

    def run(self):
        while True:
            try:
                t0 = time.time()
                dell = get_dell_system()
                fg_dell = get_dell_flightgen()
                lora = get_lora_status()
                ally = get_ally_status()
                self._history.append({
                    "t": round(time.time()), "dfg": fg_dell["completed"],
                    "afg": ally["flights"], "ls": lora["step"],
                    "dtf": fg_dell["total_files"], "atf": ally["total_files"]})
                self._history = self._history[-120:]
                with self._lock:
                    self._data = {
                        "timestamp": time.time(),
                        "uptime_min": round((time.time() - self._start_time) / 60, 1),
                        "ms": round((time.time() - t0) * 1000),
                        "hardware": self._hw,
                        "dell": dell, "flightgen_dell": fg_dell,
                        "lora": lora, "ally": ally,
                        "history": list(self._history)}
                print(f"[{time.strftime('%H:%M:%S')}] dell_fg={fg_dell['completed']} "
                      f"ally_fg={ally['flights']} lora={lora['step']}/{LORA_TOTAL_STEPS} "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[collector] Error: {e}")
            time.sleep(REFRESH_SECONDS)

    def get_data(self):
        with self._lock:
            return self._data.copy()

# ============================================================================
# HTML DASHBOARD
# ============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>FlightMind</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--dim:#8b949e;
--blue:#58a6ff;--green:#3fb950;--red:#f85149;--orange:#d29922;--purple:#bc8cff;--track:#21262d}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro','Segoe UI',system-ui,sans-serif;
background:var(--bg);color:var(--text);padding:12px;
padding-top:max(12px,env(safe-area-inset-top));
padding-bottom:max(12px,env(safe-area-inset-bottom));
-webkit-font-smoothing:antialiased}
.wrap{max-width:1200px;margin:0 auto}
.hdr{text-align:center;margin-bottom:14px}
.hdr h1{font-size:22px;font-weight:700;
background:linear-gradient(135deg,var(--blue),var(--purple));
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hdr .meta{font-size:12px;color:var(--dim);margin-top:3px}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);
margin-right:4px;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.ov{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}
.ov-item{background:var(--card);border:1px solid var(--border);border-radius:8px;
padding:10px;text-align:center}
.ov-item .v{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.ov-item .l{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.grid{display:grid;grid-template-columns:1fr;gap:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px}
.card-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.card-t{font-size:14px;font-weight:600;display:flex;align-items:center;gap:6px}
.badge{font-size:10px;padding:2px 6px;border-radius:10px;font-weight:500}
.badge.on{background:rgba(63,185,80,.15);color:var(--green)}
.badge.off{background:rgba(248,81,73,.15);color:var(--red)}
.badge.idle{background:rgba(139,148,158,.15);color:var(--dim)}
.badge.ph{background:rgba(139,148,158,.1);color:var(--dim)}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px}
.sg3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 12px}
.s .l{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px}
.s .v{font-size:16px;font-weight:600;font-variant-numeric:tabular-nums}
.s .v.sm{font-size:13px}
.pb{margin-top:10px}
.pb-lbl{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-bottom:4px}
.pb-bar{height:6px;background:var(--track);border-radius:3px;overflow:hidden}
.pb-fill{height:100%;border-radius:3px;transition:width 1s ease;min-width:2px}
.f-blue{background:var(--blue)}.f-green{background:var(--green)}
.f-orange{background:var(--orange)}.f-purple{background:var(--purple)}.f-red{background:var(--red)}
.sep{border-top:1px solid var(--border);margin:10px 0}
.chart-wrap{margin-top:10px;position:relative;height:80px}
.chart-wrap canvas{width:100%;height:100%;display:block}
.chart-lbl{font-size:10px;color:var(--dim);margin-bottom:4px}
.eta-txt{font-size:12px;color:var(--dim);margin-top:6px}
.tc{color:var(--green)}.tw{color:var(--orange)}.th{color:var(--red)}
.gpu-s{margin-top:8px;padding-top:8px;border-top:1px solid var(--border)}
.footer{text-align:center;font-size:11px;color:var(--dim);margin-top:16px;padding-bottom:8px}
/* Desktop-only sections */
.dsk{display:none}
.pred-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)}
.pred-row:last-child{border-bottom:none}
.pred-row .lbl{color:var(--dim);font-size:12px}
.pred-row .val{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}
.hw-grid{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}
.hw-grid .lbl{color:var(--dim)}
.rate-chart{height:100px;margin-top:10px}
.rate-chart canvas{width:100%;height:100%;display:block}
/* Desktop overrides */
@media(min-width:768px){
  .hdr h1{font-size:28px}
  .ov{grid-template-columns:repeat(5,1fr)}
  .grid{grid-template-columns:1fr 1fr;gap:12px}
  .span2{grid-column:span 2}
  .dsk{display:block}
  .chart-wrap{height:120px}
  .rate-chart{height:120px}
  .card{padding:18px}
  .s .v{font-size:18px}
}
@media(min-width:1024px){
  .hdr h1{font-size:32px}
  .chart-wrap{height:150px}
  .rate-chart{height:150px}
}
</style>
</head>
<body>
<div class="wrap">
<div class="hdr">
  <h1>FlightMind</h1>
  <div class="meta"><span class="dot" id="dot"></span><span id="ts">Loading...</span>
  <span class="dsk" style="display:inline" id="hw-sub"></span></div>
</div>

<div class="ov" id="ov-grid">
  <div class="ov-item"><div class="v" id="ov-fl">--</div><div class="l">Flights</div></div>
  <div class="ov-item"><div class="v" id="ov-fi">--</div><div class="l">Files</div></div>
  <div class="ov-item"><div class="v" id="ov-tk">--</div><div class="l">Est. Tokens</div></div>
  <div class="ov-item dsk"><div class="v" id="ov-active">--</div><div class="l">Active Tasks</div></div>
  <div class="ov-item dsk"><div class="v" id="ov-up">--</div><div class="l">Uptime</div></div>
</div>

<div class="grid">

<!-- LoRA Training -->
<div class="card" id="c-lora">
  <div class="card-h"><div class="card-t">LoRA Fine-tuning</div><span class="badge idle" id="lora-b">--</span></div>
  <div class="sg">
    <div class="s"><div class="l">Step</div><div class="v" id="lr-step">--</div></div>
    <div class="s"><div class="l">Loss</div><div class="v" id="lr-loss">--</div></div>
    <div class="s"><div class="l">Learning Rate</div><div class="v sm" id="lr-lr">--</div></div>
    <div class="s"><div class="l">Speed</div><div class="v" id="lr-spd">--</div></div>
  </div>
  <div class="sg dsk" style="margin-top:8px">
    <div class="s"><div class="l">Best Loss</div><div class="v sm" id="lr-best">--</div></div>
    <div class="s"><div class="l">Checkpoint</div><div class="v sm" id="lr-ckpt">--</div></div>
    <div class="s"><div class="l">ETA</div><div class="v sm" id="lr-eta">--</div></div>
    <div class="s"><div class="l">Completion</div><div class="v sm" id="lr-comp">--</div></div>
  </div>
  <div class="pb">
    <div class="pb-lbl"><span id="lr-pct">0%</span><span id="lr-gn">grad: --</span></div>
    <div class="pb-bar"><div class="pb-fill f-blue" id="lr-bar" style="width:0%"></div></div>
  </div>
  <div class="chart-wrap"><div class="chart-lbl">Loss History</div><canvas id="lc"></canvas></div>
</div>

<!-- Dell System -->
<div class="card">
  <div class="card-h"><div class="card-t">Dell 7920</div><span class="badge on">Online</span></div>
  <div class="sg">
    <div class="s"><div class="l">CPU</div><div class="v" id="d-cpu">--%</div></div>
    <div class="s"><div class="l">RAM</div><div class="v" id="d-ram">--</div></div>
  </div>
  <div class="gpu-s">
    <div class="sg">
      <div class="s"><div class="l">GPU Util</div><div class="v" id="d-gpu">--%</div></div>
      <div class="s"><div class="l">GPU Temp</div><div class="v" id="d-gt">--</div></div>
      <div class="s"><div class="l">VRAM</div><div class="v sm" id="d-vr">--</div></div>
      <div class="s"><div class="l">Disk Free</div><div class="v sm" id="d-disk">--</div></div>
    </div>
  </div>
  <div class="dsk" style="margin-top:10px">
    <div class="sep"></div>
    <div class="card-h"><div class="card-t" style="font-size:12px">Hardware</div></div>
    <div class="hw-grid" id="hw-info"></div>
  </div>
</div>

<!-- Dell Flight Gen -->
<div class="card">
  <div class="card-h"><div class="card-t">Dell Flight Gen (m0)</div></div>
  <div class="sg">
    <div class="s"><div class="l">Flights</div><div class="v" id="df-n">--</div></div>
    <div class="s"><div class="l">Rate</div><div class="v" id="df-r">--</div></div>
  </div>
  <div class="sg dsk" style="margin-top:8px">
    <div class="s"><div class="l">Landed</div><div class="v sm" id="df-land">--</div></div>
    <div class="s"><div class="l">Completion</div><div class="v sm" id="df-comp">--</div></div>
  </div>
  <div class="pb">
    <div class="pb-lbl"><span id="df-pct">0%</span><span id="df-eta">ETA: --</span></div>
    <div class="pb-bar"><div class="pb-fill f-green" id="df-bar" style="width:0%"></div></div>
  </div>
  <div class="eta-txt" id="df-files">-- files written</div>
  <div class="rate-chart dsk"><div class="chart-lbl">Flight Rate (flights completed over time)</div><canvas id="rc-dell"></canvas></div>
</div>

<!-- Ally -->
<div class="card">
  <div class="card-h"><div class="card-t">ROG Ally X</div><span class="badge idle" id="ally-b">--</span></div>
  <div class="sg">
    <div class="s"><div class="l">CPU</div><div class="v" id="a-cpu">--%</div></div>
    <div class="s"><div class="l">RAM</div><div class="v" id="a-ram">--</div></div>
  </div>
  <div class="sep"></div>
  <div class="card-h"><div class="card-t">Flight Gen (m2)</div></div>
  <div class="sg">
    <div class="s"><div class="l">Flights</div><div class="v" id="af-n">--</div></div>
    <div class="s"><div class="l">Rate</div><div class="v" id="af-r">--</div></div>
  </div>
  <div class="sg dsk" style="margin-top:8px">
    <div class="s"><div class="l">Completion</div><div class="v sm" id="af-comp">--</div></div>
    <div class="s"><div class="l">Total Files</div><div class="v sm" id="af-tf">--</div></div>
  </div>
  <div class="pb">
    <div class="pb-lbl"><span id="af-pct">0%</span><span id="af-eta">ETA: --</span></div>
    <div class="pb-bar"><div class="pb-fill f-orange" id="af-bar" style="width:0%"></div></div>
  </div>
  <div class="eta-txt" id="af-files">-- files written</div>
</div>

<!-- Predictions (desktop) -->
<div class="card dsk">
  <div class="card-h"><div class="card-t">Predictions &amp; Projections</div></div>
  <div id="pred-list">
    <div class="pred-row"><span class="lbl">LoRA completes</span><span class="val" id="p-lora">--</span></div>
    <div class="pred-row"><span class="lbl">Dell FG completes</span><span class="val" id="p-dfg">--</span></div>
    <div class="pred-row"><span class="lbl">Ally FG completes</span><span class="val" id="p-afg">--</span></div>
    <div class="pred-row"><span class="lbl">Total flights (projected)</span><span class="val" id="p-fl">--</span></div>
    <div class="pred-row"><span class="lbl">Total tokens (projected)</span><span class="val" id="p-tk">--</span></div>
    <div class="pred-row"><span class="lbl">Corpus size (projected)</span><span class="val" id="p-sz">--</span></div>
  </div>
  <div class="rate-chart"><div class="chart-lbl">Cumulative Flights Over Time</div><canvas id="rc-cum"></canvas></div>
</div>

<!-- Mac Placeholder -->
<div class="card" style="opacity:.5">
  <div class="card-h"><div class="card-t">Mac M3 Pro</div><span class="badge ph">Placeholder</span></div>
  <div class="sg">
    <div class="s"><div class="l">Status</div><div class="v sm">Not connected</div></div>
    <div class="s"><div class="l">Flight Gen (m1)</div><div class="v sm">Pending</div></div>
  </div>
</div>

</div><!-- grid -->
<div class="footer">Auto-refresh 30s | <span id="collect-ms">--</span>ms collection</div>
</div><!-- wrap -->

<script>
const $=id=>document.getElementById(id);
function fmt(n){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return ''+n}
function fmtEta(min){if(!min||min<=0)return'--';if(min<60)return Math.round(min)+'m';
  const h=Math.floor(min/60),m=Math.round(min%60);return h+'h '+m+'m'}
function fmtTime(date){return date.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}
function fmtDate(date){const m=date.getMonth()+1,d=date.getDate();
  return m+'/'+d+' '+fmtTime(date)}
function tempClass(t){return t<55?'tc':t<75?'tw':'th'}
const isDsk=window.innerWidth>=768;

function drawLineChart(canvasId,data,xKey,yKey,color,opts={}){
  const canvas=$(canvasId);if(!canvas||!data||data.length<2)return;
  const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,pad=opts.padLeft||0;
  ctx.clearRect(0,0,w,h);
  const xs=data.map(d=>d[xKey]),ys=data.map(d=>d[yKey]);
  const minX=Math.min(...xs),maxX=Math.max(...xs);
  let minY=opts.minY!=null?opts.minY:Math.min(...ys)*0.8;
  let maxY=opts.maxY!=null?opts.maxY:Math.max(...ys)*1.1;
  if(maxY===minY){maxY=minY+1}
  const rX=maxX-minX||1,rY=maxY-minY;
  // Grid
  ctx.strokeStyle='#30363d';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){const y=h*i/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w,y);ctx.stroke()}
  // Line
  ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
  data.forEach((d,i)=>{
    const x=pad+(d[xKey]-minX)/rX*(w-pad);
    const y=h-(d[yKey]-minY)/rY*h;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });ctx.stroke();
  // Fill under
  if(opts.fill){
    const last=data[data.length-1];
    const lx=pad+(last[xKey]-minX)/rX*(w-pad);
    ctx.lineTo(lx,h);ctx.lineTo(pad,h);ctx.closePath();
    ctx.fillStyle=opts.fill;ctx.fill();
  }
  // Best marker
  if(opts.markBest){
    const best=data.reduce((a,b)=>a[yKey]<b[yKey]?a:b);
    const bx=pad+(best[xKey]-minX)/rX*(w-pad),by=h-(best[yKey]-minY)/rY*h;
    ctx.fillStyle='#3fb950';ctx.beginPath();ctx.arc(bx,by,3,0,Math.PI*2);ctx.fill();
  }
  // Labels
  ctx.fillStyle='#8b949e';ctx.font='9px system-ui';
  ctx.fillText(minY.toFixed(opts.decimals||3),2,h-2);
  ctx.fillText(maxY.toFixed(opts.decimals||3),2,10);
  if(opts.rightLabel){ctx.textAlign='right';ctx.fillText(opts.rightLabel,w-2,h-2);ctx.textAlign='left'}
}

async function refresh(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    $('ts').textContent='Updated '+new Date(d.timestamp*1000).toLocaleTimeString();
    $('collect-ms').textContent=d.ms||'--';

    // Overview
    const dfc=d.flightgen_dell?.completed||0, afc=d.ally?.flights||0;
    const tFlights=dfc+afc;
    const tFiles=(d.flightgen_dell?.total_files||0)+(d.ally?.total_files||0);
    const tTokens=Math.round(tFiles*200);
    $('ov-fl').textContent=fmt(tFlights);
    $('ov-fi').textContent=fmt(tFiles);
    $('ov-tk').textContent=fmt(tTokens);
    let active=0;
    if(d.lora?.active)active++;
    if((d.flightgen_dell?.rate||0)>0)active++;
    if(d.ally?.online&&(d.ally?.rate||0)>0)active++;
    if($('ov-active'))$('ov-active').textContent=active;
    if($('ov-up'))$('ov-up').textContent=fmtEta(d.uptime_min);

    // Hardware subtitle
    if(d.hardware&&$('hw-sub')){
      $('hw-sub').textContent=' | '+d.hardware.hostname;
    }

    // LoRA
    const l=d.lora||{};
    const lb=$('lora-b');
    if(l.active){lb.textContent='Training';lb.className='badge on'}
    else if(l.step>=l.total_steps){lb.textContent='Complete';lb.className='badge on'}
    else{lb.textContent='Idle';lb.className='badge idle'}
    $('lr-step').textContent=l.step+'/'+l.total_steps;
    $('lr-loss').textContent=l.loss?l.loss.toFixed(4):'--';
    $('lr-lr').textContent=l.lr?l.lr.toExponential(2):'--';
    $('lr-spd').textContent=l.tok_s?l.tok_s+' tok/s':'--';
    $('lr-pct').textContent=l.pct?l.pct.toFixed(1)+'%':'0%';
    $('lr-gn').textContent='grad: '+(l.grad_norm?l.grad_norm.toFixed(2):'--');
    $('lr-bar').style.width=(l.pct||0)+'%';
    // Desktop extras
    if($('lr-best'))$('lr-best').textContent=l.best_loss<999?l.best_loss.toFixed(4):'--';
    if($('lr-ckpt'))$('lr-ckpt').textContent='step '+l.checkpoint_step;
    // LoRA ETA prediction
    if(l.active&&l.tok_s>0&&l.step>0){
      const hist=d.history||[];
      if(hist.length>=2){
        const h0=hist[0],h1=hist[hist.length-1];
        const dt=(h1.t-h0.t)/60,ds=h1.ls-h0.ls;
        if(ds>0&&dt>0){
          const stepsPerMin=ds/dt;
          const remain=l.total_steps-l.step;
          const etaMin=remain/stepsPerMin;
          if($('lr-eta'))$('lr-eta').textContent=fmtEta(etaMin);
          const comp=new Date(Date.now()+etaMin*60000);
          if($('lr-comp'))$('lr-comp').textContent=fmtDate(comp);
        }
      }
    }
    if(l.loss_history&&l.loss_history.length>1){
      drawLineChart('lc',l.loss_history,'step','loss','#f85149',{markBest:true,fill:'rgba(248,81,73,0.05)'});
    }

    // Dell system
    const ds=d.dell||{};
    $('d-cpu').textContent=(ds.cpu_pct||0)+'%';
    $('d-ram').textContent=(ds.ram_used_gb||0)+'/'+(ds.ram_total_gb||0)+' GB';
    $('d-gpu').textContent=(ds.gpu_util||0)+'%';
    const gt=ds.gpu_temp||0;
    $('d-gt').innerHTML='<span class="'+tempClass(gt)+'">'+gt+'&deg;C</span>';
    $('d-vr').textContent=(ds.gpu_vram_used||0)+'/'+(ds.gpu_vram_total||0)+' MB';
    $('d-disk').textContent=(ds.disk_free_gb||'--')+' GB';
    // Hardware info
    if(d.hardware&&$('hw-info')){
      const hw=d.hardware;
      let html='';
      if(hw.cpu_model)html+='<span class="lbl">CPU</span><span>'+hw.cpu_model+'</span>';
      if(hw.gpu_model)html+='<span class="lbl">GPU</span><span>'+hw.gpu_model+'</span>';
      if(hw.gpu_driver)html+='<span class="lbl">Driver</span><span>'+hw.gpu_driver+'</span>';
      html+='<span class="lbl">Cores</span><span>'+hw.cpu_cores+'</span>';
      if(hw.disk_total_gb)html+='<span class="lbl">Disk</span><span>'+hw.disk_total_gb+' GB total</span>';
      if(hw.os)html+='<span class="lbl">OS</span><span style="font-size:11px">'+hw.os+'</span>';
      $('hw-info').innerHTML=html;
    }

    // Dell flight gen
    const df=d.flightgen_dell||{};
    $('df-n').textContent=(df.completed||0)+'/'+df.total;
    $('df-r').textContent=(df.rate||0)+' fl/s';
    $('df-pct').textContent=(df.pct||0).toFixed(1)+'%';
    $('df-eta').textContent='ETA: '+fmtEta(df.eta_min);
    $('df-bar').style.width=(df.pct||0)+'%';
    $('df-files').textContent=(df.total_files||0)+' files written';
    if($('df-land'))$('df-land').textContent=df.landed||'--';
    if(df.eta_min>0&&$('df-comp')){
      $('df-comp').textContent=fmtDate(new Date(Date.now()+df.eta_min*60000));
    }

    // Ally
    const a=d.ally||{};
    const ab=$('ally-b');
    if(a.online){ab.textContent='Online';ab.className='badge on'}
    else{ab.textContent='Offline';ab.className='badge off'}
    $('a-cpu').textContent=a.online?(a.cpu_pct||0)+'%':'--';
    $('a-ram').textContent=a.online?(a.ram_used_gb||0)+'/'+(a.ram_total_gb||0)+' GB':'--';
    $('af-n').textContent=(a.flights||0)+'/'+a.total;
    $('af-r').textContent=(a.rate||0)+' fl/s';
    $('af-pct').textContent=(a.pct||0).toFixed(1)+'%';
    $('af-eta').textContent='ETA: '+fmtEta(a.eta_min);
    $('af-bar').style.width=(a.pct||0)+'%';
    $('af-files').textContent=(a.total_files||0)+' files written';
    if($('af-tf'))$('af-tf').textContent=a.total_files||'--';
    if(a.eta_min>0&&$('af-comp')){
      $('af-comp').textContent=fmtDate(new Date(Date.now()+a.eta_min*60000));
    }

    // Predictions
    const projFlights=DELL_TOTAL+ALLY_TOTAL;
    const projTokens=projFlights*5*200;
    const projSizeMB=Math.round(projTokens*4/1e6);
    if($('p-fl'))$('p-fl').textContent=fmt(projFlights)+' flights';
    if($('p-tk'))$('p-tk').textContent=fmt(projTokens)+' tokens';
    if($('p-sz'))$('p-sz').textContent='~'+projSizeMB+' MB';
    if(df.eta_min>0&&$('p-dfg'))$('p-dfg').textContent=fmtDate(new Date(Date.now()+df.eta_min*60000));
    if(a.eta_min>0&&$('p-afg'))$('p-afg').textContent=fmtDate(new Date(Date.now()+a.eta_min*60000));

    // Rate charts (desktop)
    const hist=d.history||[];
    if(hist.length>=2){
      // Cumulative flights chart
      const cumData=hist.map(h=>({t:h.t,flights:h.dfg+h.afg}));
      drawLineChart('rc-cum',cumData,'t','flights','#58a6ff',
        {fill:'rgba(88,166,255,0.08)',decimals:0,rightLabel:cumData[cumData.length-1].flights+' flights'});
      // Dell rate chart
      const dellHist=hist.map(h=>({t:h.t,n:h.dfg}));
      drawLineChart('rc-dell',dellHist,'t','n','#3fb950',
        {fill:'rgba(63,185,80,0.08)',decimals:0,minY:0});
    }
  }catch(e){console.error('refresh failed',e)}
}
const DELL_TOTAL=20000,ALLY_TOTAL=15000;
refresh();setInterval(refresh,30000);
</script>
</body>
</html>"""

# ============================================================================
# HTTP SERVER
# ============================================================================

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    collector = None

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = self.collector.get_data() if self.collector else {}
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"])

    print("Collecting hardware info...")
    hw = get_hardware_info()
    print(f"  {hw.get('cpu_model', 'Unknown CPU')}")
    print(f"  {hw.get('gpu_model', 'No GPU')}")

    collector = StatusCollector(hw)
    collector.start()

    DashboardHandler.collector = collector
    server = http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler)

    ts_ip = ""
    try:
        r = subprocess.run(["tailscale", "ip", "-4"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ts_ip = r.stdout.strip()
    except Exception:
        pass

    print(f"\nFlightMind Dashboard")
    print(f"  Local:     http://localhost:{PORT}")
    if ts_ip:
        print(f"  Tailscale: http://{ts_ip}:{PORT}")
    print(f"  Mobile:  compact cards")
    print(f"  Desktop: 2-col grid + charts + predictions")
    print(f"\nCtrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()

if __name__ == "__main__":
    main()
