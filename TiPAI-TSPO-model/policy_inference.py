import argparse
import json
from typing import Any, Dict, List, Optional

def tau_policy(t_norm: float) -> float:
    return 0.40 + 0.25 * t_norm


def tau_faithfulness(t_norm: float) -> float:
    if t_norm < 0.85:
        return 0.30 + 0.30 * t_norm
    return 0.55 - 0.10 * (t_norm - 0.85) / 0.15


def compute_guard_utility(p_i: float, f_i: float, b_i: float, t_norm: float) -> float:
    policy_ok = float(p_i >= tau_policy(t_norm))
    faith_ok = float(f_i >= tau_faithfulness(t_norm))
    return policy_ok * faith_ok * b_i


def _pick_float(d: Dict[str, Any], keys: List[str], field_name: str) -> float:
    for k in keys:
        if k in d:
            return float(d[k])
    raise ValueError(f"Missing field '{field_name}'. Expected one of {keys}.")


def _score_one(candidate: Dict[str, Any], default_t_norm: Optional[float]) -> Dict[str, Any]:
    t_norm = float(candidate.get("t_norm", default_t_norm))
    if t_norm is None:
        raise ValueError("t_norm is required (globally or per candidate).")

    p_r = _pick_float(candidate, ["P_R", "P_i", "policy_safe"], "P_R")
    f_r = _pick_float(candidate, ["F_R", "F_i", "faithfulness"], "F_R")
    b_i = _pick_float(candidate, ["B_i", "B", "seam_quality"], "B_i")

    utility = float(compute_guard_utility(p_r, f_r, b_i, t_norm))
    p_thr = float(tau_policy(t_norm))
    f_thr = float(tau_faithfulness(t_norm))

    return {
        "P_R": p_r,
        "F_R": f_r,
        "B_i": b_i,
        "t_norm": t_norm,
        "tau_policy": p_thr,
        "tau_faithfulness": f_thr,
        "policy_gate_pass": bool(p_r >= p_thr),
        "faithfulness_gate_pass": bool(f_r >= f_thr),
        "utility": utility,
    }


def _load_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            candidates = payload.get("candidates", payload.get("items"))
            t_norm = payload.get("t_norm", args.t_norm)
        elif isinstance(payload, list):
            candidates = payload
            t_norm = args.t_norm
        else:
            raise ValueError("JSON input must be a dict or list.")

        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValueError("Input JSON must include non-empty 'candidates' list.")

        return {"candidates": candidates, "t_norm": t_norm}

    if args.p_r is None or args.f_r is None or args.b_i is None or args.t_norm is None:
        raise ValueError(
            "For single-candidate mode, provide --p-r --f-r --b-i --t-norm."
        )

    single = [{"P_R": args.p_r, "F_R": args.f_r, "B_i": args.b_i, "t_norm": args.t_norm}]
    return {"candidates": single, "t_norm": args.t_norm}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Training-policy inference for reward decision: "
            "compute utility and pick winning candidate."
        )
    )

    p.add_argument(
        "--input-json",
        default=None,
        help=(
            "Path to JSON with either {'t_norm': float, 'candidates': [...]} "
            "or a plain list of candidate dicts."
        ),
    )

    p.add_argument("--p-r", type=float, default=None, help="Candidate P_R (policy_safe on region)")
    p.add_argument("--f-r", type=float, default=None, help="Candidate F_R (faithfulness on region)")
    p.add_argument("--b-i", type=float, default=None, help="Candidate B_i (seam quality)")
    p.add_argument("--t-norm", type=float, default=None, help="Normalized diffusion timestep in [0, 1]")

    return p


def main() -> None:
    args = build_parser().parse_args()
    data = _load_candidates(args)

    scored = [_score_one(c, data.get("t_norm")) for c in data["candidates"]]
    utilities = [x["utility"] for x in scored]

    winner_idx = max(range(len(utilities)), key=lambda i: utilities[i])
    best_utility = float(utilities[winner_idx])
    accepted = bool(best_utility > 0.0)

    result = {
        "num_candidates": len(scored),
        "utilities": utilities,
        "winner_idx": winner_idx,
        "winner": scored[winner_idx],
        "accepted": accepted,
        "scored_candidates": scored,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
