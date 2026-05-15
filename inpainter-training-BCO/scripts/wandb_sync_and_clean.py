#!/usr/bin/env python3
"""Sync local W&B runs and optionally clean local W&B files.

Typical usage:
  python scripts/wandb_sync_and_clean.py --sync
  python scripts/wandb_sync_and_clean.py --sync --clean --yes
  python scripts/wandb_sync_and_clean.py --sync --clean --yes --wandb-dir ./wandb

Notes:
- This script uses `wandb sync <run_dir>` for each discovered run directory.
- Cleanup only deletes local files under the chosen wandb directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync and clean local W&B run files")
    parser.add_argument("--wandb-dir", default="wandb", help="Path to local wandb directory (default: wandb)")
    parser.add_argument("--sync", action="store_true", help="Sync all local run directories to W&B")
    parser.add_argument("--clean", action="store_true", help="Delete local wandb files after optional sync")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt for cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them")
    return parser.parse_args()


def find_run_dirs(wandb_dir: Path) -> list[Path]:
    if not wandb_dir.exists():
        return []

    run_dirs: list[Path] = []
    for child in sorted(wandb_dir.iterdir()):
        if not child.is_dir():
            continue
        # Common local run folder patterns.
        if child.name.startswith("run-") or child.name.startswith("offline-run-"):
            run_dirs.append(child)
    return run_dirs


def run_sync(run_dir: Path, dry_run: bool) -> bool:
    cmd = ["wandb", "sync", str(run_dir)]
    print(f"[sync] {' '.join(cmd)}")
    if dry_run:
        return True

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as err:
        print(f"[sync] failed for {run_dir}: {err}", file=sys.stderr)
        return False


def cleanup_wandb_dir(wandb_dir: Path, keep_failed: set[Path], dry_run: bool) -> None:
    # Remove run dirs that were synced successfully.
    for run_dir in find_run_dirs(wandb_dir):
        if run_dir in keep_failed:
            print(f"[clean] keeping failed sync run: {run_dir}")
            continue
        print(f"[clean] remove dir: {run_dir}")
        if not dry_run:
            shutil.rmtree(run_dir, ignore_errors=True)

    # Remove local helper symlink/folder and debug logs.
    for name in ["latest-run", "debug.log", "debug-internal.log", "debug-cli.root.log"]:
        target = wandb_dir / name
        if not target.exists() and not target.is_symlink():
            continue
        print(f"[clean] remove: {target}")
        if dry_run:
            continue
        if target.is_symlink() or target.is_file():
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        elif target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


def main() -> int:
    args = parse_args()
    wandb_dir = Path(args.wandb_dir).resolve()

    if not args.sync and not args.clean:
        print("Nothing to do. Use --sync and/or --clean.")
        return 2

    if shutil.which("wandb") is None and args.sync:
        print("`wandb` CLI is not available in PATH.", file=sys.stderr)
        return 1

    run_dirs = find_run_dirs(wandb_dir)
    print(f"Found {len(run_dirs)} local run directories in {wandb_dir}")

    failed_sync: set[Path] = set()
    if args.sync:
        for run_dir in run_dirs:
            ok = run_sync(run_dir, dry_run=args.dry_run)
            if not ok:
                failed_sync.add(run_dir)

        synced_count = len(run_dirs) - len(failed_sync)
        print(f"Sync complete: {synced_count}/{len(run_dirs)} succeeded")

    if args.clean:
        if failed_sync:
            print("Some runs failed to sync; they will be kept locally.")

        if not args.yes and not args.dry_run:
            answer = input("Delete local W&B files now? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Cleanup skipped.")
                return 0

        cleanup_wandb_dir(wandb_dir, keep_failed=failed_sync, dry_run=args.dry_run)
        print("Cleanup complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
