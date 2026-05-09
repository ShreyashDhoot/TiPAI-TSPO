import argparse
import json
from dataclasses import asdict
from typing import List

import torch

from src.config import DEVICE, STATE_DIM
from src.models import TSPOPolicy
from src.utils.helpers import denorm


def parse_state_vector(args: argparse.Namespace) -> torch.Tensor:
    if bool(args.state) == bool(args.state_json):
        raise ValueError("Provide exactly one of --state or --state-json.")

    if args.state:
        vals = [float(x.strip()) for x in args.state.split(",") if x.strip()]
    else:
        with open(args.state_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "state" in payload:
            payload = payload["state"]

        if not isinstance(payload, list):
            raise ValueError("State JSON must be a list or an object with key 'state'.")

        vals = [float(x) for x in payload]

    if len(vals) != STATE_DIM:
        raise ValueError(f"Expected state vector length {STATE_DIM}, got {len(vals)}.")

    return torch.tensor(vals, dtype=torch.float32)


def load_policy(checkpoint_path: str, device: torch.device) -> TSPOPolicy:
    policy = TSPOPolicy(state_dim=STATE_DIM).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Training saves policy.state_dict(), but allow dict wrappers too.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "policy" in ckpt:
        state_dict = ckpt["policy"]
    else:
        state_dict = ckpt

    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def deterministic_knob(policy: TSPOPolicy, state: torch.Tensor) -> dict:
    from src.config import KNOB_BOUNDS

    with torch.no_grad():
        mean, _log_std, seed_logits = policy(state.unsqueeze(0))

    raw = mean[0].clamp(0, 1).tolist()
    seed_bucket = int(torch.argmax(seed_logits[0]).item())

    out = {
        "cfg_scale": denorm(raw[0], *KNOB_BOUNDS["cfg_scale"]),
        "mask_dilation": denorm(raw[1], *KNOB_BOUNDS["mask_dilation"]),
        "mask_feather": denorm(raw[2], *KNOB_BOUNDS["mask_feather"]),
        "noise_jitter": denorm(raw[3], *KNOB_BOUNDS["noise_jitter"]),
        "inversion_depth": max(1, int(round(denorm(raw[4], *KNOB_BOUNDS["inversion_depth"]))),),
        "seed_offset": seed_bucket * 100,
        "raw_cont": raw,
        "log_prob": None,
    }
    return out


def run_inference(
    policy: TSPOPolicy,
    state: torch.Tensor,
    num_suggestions: int,
    deterministic: bool,
) -> List[dict]:
    if deterministic:
        return [deterministic_knob(policy, state)]

    with torch.no_grad():
        knob_sets = policy.sample_knobs(state.to(next(policy.parameters()).device), N=num_suggestions)

    return [asdict(k) for k in knob_sets]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tournament policy inference: state features -> knob suggestions"
    )
    p.add_argument("--checkpoint", required=True, help="Path to TSPO policy checkpoint (.pth)")
    p.add_argument(
        "--state",
        default=None,
        help=f"Comma-separated state vector of length {STATE_DIM}",
    )
    p.add_argument(
        "--state-json",
        default=None,
        help="Path to JSON file containing a list under root or under key 'state'",
    )
    p.add_argument(
        "--num-suggestions",
        type=int,
        default=1,
        help="Number of sampled suggestions (ignored when --deterministic is set)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use policy mean + argmax seed bucket instead of stochastic sampling",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for stochastic sampling")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.num_suggestions < 1:
        raise ValueError("--num-suggestions must be >= 1")

    if args.device == "auto":
        device = DEVICE
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if str(device) == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    state = parse_state_vector(args).to(device)
    policy = load_policy(args.checkpoint, device)

    outputs = run_inference(
        policy=policy,
        state=state,
        num_suggestions=args.num_suggestions,
        deterministic=args.deterministic,
    )

    result = {
        "state_dim": STATE_DIM,
        "num_outputs": len(outputs),
        "deterministic": bool(args.deterministic),
        "outputs": outputs,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
