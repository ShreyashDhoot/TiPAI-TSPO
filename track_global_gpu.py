#!/usr/bin/env python3
"""Track global GPU usage over time and optionally write samples to CSV.

This reports an aggregate view across all visible NVIDIA GPUs:
- average GPU utilization
- total/used memory and overall memory utilization
- average temperature
- total power draw

Examples:
  python track_global_gpu.py
  python track_global_gpu.py --interval 1 --duration 300 --output gpu_global_usage.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import signal
import subprocess
import sys
import time
from statistics import mean
from typing import Dict, List


STOP = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track global NVIDIA GPU usage over time")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between samples (default: 2.0)")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total seconds to monitor; 0 means run until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional CSV file path. If omitted, only prints to console.",
    )
    return parser.parse_args()


def require_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        print("ERROR: nvidia-smi not found. This script needs NVIDIA drivers/tools.", file=sys.stderr)
        sys.exit(1)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def sample_gpus() -> Dict[str, float]:
    query = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)

    gpu_utils: List[float] = []
    mem_utils: List[float] = []
    mem_used: List[float] = []
    mem_total: List[float] = []
    temps: List[float] = []
    power_draw: List[float] = []

    for raw_line in out.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) != 6:
            continue

        gpu_utils.append(_to_float(parts[0]))
        mem_utils.append(_to_float(parts[1]))
        mem_used.append(_to_float(parts[2]))
        mem_total.append(_to_float(parts[3]))
        temps.append(_to_float(parts[4]))
        power_draw.append(_to_float(parts[5]))

    total_used = sum(mem_used)
    total_total = sum(mem_total)
    mem_util_percent = (total_used / total_total * 100.0) if total_total > 0 else 0.0

    return {
        "gpu_count": float(len(gpu_utils)),
        "avg_gpu_util_percent": mean(gpu_utils) if gpu_utils else 0.0,
        "avg_mem_util_percent": mean(mem_utils) if mem_utils else 0.0,
        "total_mem_used_mb": total_used,
        "total_mem_total_mb": total_total,
        "mem_util_percent": mem_util_percent,
        "avg_temp_c": mean(temps) if temps else 0.0,
        "total_power_w": sum(power_draw),
    }


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        print("ERROR: --interval must be > 0", file=sys.stderr)
        return 2

    require_nvidia_smi()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    writer = None
    csv_file = None

    fieldnames = [
        "timestamp",
        "gpu_count",
        "avg_gpu_util_percent",
        "avg_mem_util_percent",
        "total_mem_used_mb",
        "total_mem_total_mb",
        "mem_util_percent",
        "avg_temp_c",
        "total_power_w",
    ]

    if args.output:
        csv_file = open(args.output, "w", newline="", encoding="ascii")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    start = time.time()
    print("Tracking global GPU usage. Press Ctrl+C to stop.")
    print("timestamp | gpus | avg util% | mem used/total MB | mem util% | avg temp C | total power W")

    try:
        while not STOP:
            now = dt.datetime.now().isoformat(timespec="seconds")
            row = {"timestamp": now, **sample_gpus()}

            print(
                f"{now} | {int(row['gpu_count'])} | {row['avg_gpu_util_percent']:.1f} | "
                f"{row['total_mem_used_mb']:.0f}/{row['total_mem_total_mb']:.0f} | "
                f"{row['mem_util_percent']:.1f} | {row['avg_temp_c']:.1f} | {row['total_power_w']:.1f}"
            )

            if writer:
                writer.writerow(row)
                csv_file.flush()

            if args.duration > 0 and (time.time() - start) >= args.duration:
                break

            time.sleep(args.interval)
    finally:
        if csv_file:
            csv_file.close()

    print("Tracking stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())