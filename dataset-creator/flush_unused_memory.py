#!/usr/bin/env python3
"""
Flush reclaimable Linux memory (cache) and monitor RAM usage.

What this can do:
- Show current memory usage from /proc/meminfo
- Optionally drop reclaimable caches (requires root)
- Optionally trigger memory compaction (requires root)

What this cannot do:
- Force another running Python process to release its private heap/objects
- Free VRAM used by another process
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict


def read_meminfo() -> Dict[str, int]:
    data: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if not parts:
                continue
            # meminfo values are in kB
            try:
                data[key] = int(parts[0]) * 1024
            except ValueError:
                continue
    return data


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f}{units[idx]}"


def print_summary(prefix: str = "") -> None:
    m = read_meminfo()
    total = m.get("MemTotal", 0)
    avail = m.get("MemAvailable", 0)
    free = m.get("MemFree", 0)
    cached = m.get("Cached", 0)
    s_reclaimable = m.get("SReclaimable", 0)
    buffers = m.get("Buffers", 0)

    used_est = max(total - avail, 0)
    reclaimable_est = cached + s_reclaimable + buffers

    msg = (
        f"{prefix}MemTotal={fmt_bytes(total)} "
        f"MemUsed(est)={fmt_bytes(used_est)} "
        f"MemAvailable={fmt_bytes(avail)} "
        f"MemFree={fmt_bytes(free)} "
        f"Reclaimable(est)={fmt_bytes(reclaimable_est)}"
    )
    print(msg)


def require_linux() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This script supports Linux only.")


def is_root() -> bool:
    return os.geteuid() == 0


def write_proc(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def drop_caches(level: int) -> None:
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")

    subprocess.run(["sync"], check=True)
    write_proc("/proc/sys/vm/drop_caches", str(level))


def compact_memory() -> None:
    write_proc("/proc/sys/vm/compact_memory", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flush reclaimable Linux memory and monitor RAM.")
    p.add_argument(
        "--drop-caches",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="0=do nothing, 1=pagecache, 2=dentries+inodes, 3=both pagecache and dentries+inodes.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Trigger kernel memory compaction (root only).",
    )
    p.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="If >0, print memory summary every N seconds.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of watch iterations (0 means infinite when watch-seconds > 0).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for cleanup operations.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_linux()

    print_summary(prefix="[BEFORE] ")

    wants_cleanup = args.drop_caches > 0 or args.compact
    if wants_cleanup:
        if not is_root():
            raise PermissionError(
                "Cleanup operations require root. Re-run with sudo, for example:\n"
                "  sudo python dataset-creator/flush_unused_memory.py --drop-caches 3 --compact --yes"
            )

        if not args.yes:
            print("About to run kernel memory cleanup operations.")
            ans = input("Continue? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return

        if args.drop_caches > 0:
            print(f"Dropping caches: level={args.drop_caches}")
            drop_caches(args.drop_caches)

        if args.compact:
            print("Triggering memory compaction")
            compact_memory()

        print_summary(prefix="[AFTER ] ")

    if args.watch_seconds > 0:
        count = 0
        while True:
            if args.iterations > 0 and count >= args.iterations:
                break
            time.sleep(args.watch_seconds)
            print_summary(prefix="[WATCH ] ")
            count += 1


if __name__ == "__main__":
    main()
