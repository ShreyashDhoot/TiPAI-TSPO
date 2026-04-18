#!/usr/bin/env python3
"""Track GPU usage over time and optionally write samples to CSV.

Examples:
  python scripts/track_gpu.py
  python scripts/track_gpu.py --interval 1 --duration 300 --output gpu_usage.csv
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
from typing import Dict, List


STOP = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track NVIDIA GPU usage over time")
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


def sample_gpus() -> List[Dict[str, str]]:
    query = (
        "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)

    rows: List[Dict[str, str]] = []
    for raw_line in out.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) != 8:
            continue

        rows.append(
            {
                "gpu_index": parts[0],
                "gpu_name": parts[1],
                "gpu_util_percent": parts[2],
                "mem_util_percent": parts[3],
                "mem_used_mb": parts[4],
                "mem_total_mb": parts[5],
                "temp_c": parts[6],
                "power_w": parts[7],
            }
        )
    return rows


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
        "gpu_index",
        "gpu_name",
        "gpu_util_percent",
        "mem_util_percent",
        "mem_used_mb",
        "mem_total_mb",
        "temp_c",
        "power_w",
    ]

    if args.output:
        csv_file = open(args.output, "w", newline="", encoding="ascii")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    start = time.time()
    print("Tracking started. Press Ctrl+C to stop.")
    print("timestamp | gpu | util% | mem used/total MB | mem util% | temp C | power W")

    try:
        while not STOP:
            now = dt.datetime.now().isoformat(timespec="seconds")
            rows = sample_gpus()

            for row in rows:
                row_with_ts = {"timestamp": now, **row}
                print(
                    f"{now} | {row['gpu_index']} | {row['gpu_util_percent']} | "
                    f"{row['mem_used_mb']}/{row['mem_total_mb']} | {row['mem_util_percent']} | "
                    f"{row['temp_c']} | {row['power_w']}"
                )
                if writer:
                    writer.writerow(row_with_ts)

            if csv_file:
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
