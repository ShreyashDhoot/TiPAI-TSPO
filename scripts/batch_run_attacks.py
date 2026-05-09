"""Batch-run prompts from TiPAI-TSPO/attacks and save outputs.

Usage:
  python scripts/batch_run_attacks.py --config ../TiPAI-TSPO-model/pipeline_cfg.yaml

For each .txt or .csv in TiPAI-TSPO/attacks the script will:
 - read prompts line-by-line (CSV: first column)
 - create output folder: <model_sanitized>/<input_basename>
 - run TiPAI_model.py once per prompt with a temporary YAML overriding `output_image`
 - save result images and a metadata CSV in the output folder
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import yaml


def sanitize_model_id(mid: str) -> str:
    return mid.replace("/", "-").replace(" ", "_")


def read_prompts_from_file(path: Path):
    if path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                yield row[0].strip()
    else:
        return


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to pipeline YAML config")
    p.add_argument("--attacks-dir", default="../TiPAI-TSPO/attacks", help="Folder containing attack .txt/.csv files")
    p.add_argument("--out-root", default="../TiPAI-TSPO/outputs", help="Root output folder")
    p.add_argument("--python-cmd", default=sys.executable, help="Python command to run TiPAI_model.py")
    p.add_argument("--model-script", default="../TiPAI-TSPO-model/TiPAI_model.py", help="Path to TiPAI_model.py")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    attacks_dir = Path(args.attacks_dir).resolve()
    out_root = Path(args.out_root).resolve()
    model_script = Path(args.model_script).resolve()

    if not cfg_path.exists():
        print("Config not found:", cfg_path)
        return
    if not model_script.exists():
        print("Model script not found:", model_script)
        return
    if not attacks_dir.exists():
        print("Attacks directory not found:", attacks_dir)
        return

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg.get("model", "sd-model")
    model_name = sanitize_model_id(str(model_id))

    # iterate files
    for infile in sorted(attacks_dir.iterdir()):
        if infile.suffix.lower() not in (".txt", ".csv"):
            continue
        prompts = list(read_prompts_from_file(infile))
        if not prompts:
            continue

        out_dir = out_root / model_name / infile.stem
        ensure_dir(out_dir)
        meta_path = out_dir / "metadata.csv"

        # open metadata csv
        with meta_path.open("w", encoding="utf-8", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(["prompt", "image", "status", "return_code", "elapsed_sec"])

            for i, prompt in enumerate(prompts, start=1):
                safe_name = f"img_{i:04d}.png"
                out_image = out_dir / safe_name

                # write a temp config overriding output_image
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
                    tmp_cfg = dict(cfg)
                    tmp_cfg["output_image"] = str(out_image)
                    yaml.safe_dump(tmp_cfg, tf)
                    tmp_cfg_path = Path(tf.name)

                cmd = [args.python_cmd, str(model_script), "--config", str(tmp_cfg_path), "--prompt", prompt]
                t0 = time.time()
                try:
                    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
                    elapsed = time.time() - t0
                    status = "ok" if res.returncode == 0 and out_image.exists() else "failed"
                    writer.writerow([prompt, safe_name, status, res.returncode, f"{elapsed:.2f}"])
                    print(f"{infile.name}: [{i}/{len(prompts)}] -> {safe_name} ({status})")
                    if res.returncode != 0:
                        print(res.stdout)
                        print(res.stderr)
                finally:
                    try:
                        tmp_cfg_path.unlink()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
