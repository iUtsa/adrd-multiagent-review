"""Cost estimate for the final runs (task 8.4).

Scales one finished run's token use to the target size and prices it.

    # After a real calibration run (measured tokens and models):
    python -m adrd_review.eval.cost data/runs/E4_gemini/<timestamp> --articles 228 --repeats 3

    # Before any real run: take the call pattern and prompt sizes from a fake run and
    # price them as the models of a Gemini config
    python -m adrd_review.eval.cost data/runs/E4_fake/<timestamp> --as-config configs/runs/E4_gemini.yaml --articles 228 --repeats 3

Prices: configs/prices.yaml. The fake model's input tokens approximate real prompt
sizes (characters / 4); its output is tiny, so fake calls are priced with
--fake-output-tokens (default 400: rationale, quotes and Gemini thinking tokens).
Treat pre-run estimates as rough; re-estimate from a real calibration run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from adrd_review.graph.runner import RunConfig, load_run_config


def role_models(cfg: RunConfig) -> dict[str, list[str]]:
    """Which model labels each role's calls go to under a run config."""
    panel = cfg.reviewer_configs()
    out = {
        "triage": [cfg.roles["triage"].model.label],
        "extraction": [cfg.roles["extraction"].model.label],
        "screening": [r.model.label for r in panel],
        "debate": [r.model.label for r in panel],
        "annotation": [(r.annotation_model or r.model).label for r in panel],
    }
    if cfg.adjudicator:
        out["adjudicator"] = [cfg.adjudicator.label]
    if cfg.verifier and cfg.verifier.model:
        out["verifier"] = [cfg.verifier.model.label]
    return out


def estimate(run_dir: Path, prices: dict, articles: int, repeats: int, as_config: RunConfig | None = None,
             fake_output_tokens: int = 400) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    n = len(meta["pmids"])
    usage = [json.loads(line) for line in (run_dir / "usage.jsonl").read_text().splitlines()]
    mapping = role_models(as_config) if as_config else None

    by_role: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tin": 0, "tout": 0, "usd": 0.0, "models": set()})
    missing: set[str] = set()
    for u in usage:
        role = u["role"]
        models = mapping.get(role, [u["model"]]) if mapping else [u["model"]]
        tout = fake_output_tokens if u["model"].startswith("fake:") else u["output_tokens"]
        rates = [prices.get(m) for m in models]
        if any(r is None for r in rates):
            missing |= {m for m, r in zip(models, rates) if r is None}
            continue
        pin = sum(r["input"] for r in rates) / len(rates)
        pout = sum(r["output"] for r in rates) / len(rates)
        b = by_role[role]
        b["calls"] += 1
        b["tin"] += u["input_tokens"]
        b["tout"] += tout
        b["usd"] += (u["input_tokens"] * pin + tout * pout) / 1e6
        b["models"] |= set(models)
    if missing:
        raise SystemExit(f"No price in configs/prices.yaml for: {sorted(missing)}")

    scale = articles * repeats / n
    rows = []
    for role, b in sorted(by_role.items(), key=lambda kv: -kv[1]["usd"]):
        rows.append({
            "role": role,
            "models": ", ".join(sorted(b["models"])),
            "calls_per_article": round(b["calls"] / n, 2),
            "input_tokens_per_call": round(b["tin"] / b["calls"]),
            "output_tokens_per_call": round(b["tout"] / b["calls"]),
            "usd_per_article": round(b["usd"] / n, 5),
            "usd_projected": round(b["usd"] * scale, 2),
        })
    no_free = sorted({m for b in by_role.values() for m in b["models"] if not prices[m].get("free_tier", True)})
    return {
        "source_run": str(run_dir),
        "priced_as": str(as_config.name) if as_config else "run's own models",
        "source_articles": n,
        "target_articles": articles,
        "repeats": repeats,
        "estimated_from_fake_model": any(u["model"].startswith("fake:") for u in usage),
        "total_usd": round(sum(r["usd_projected"] for r in rows), 2),
        "by_role": rows,
        "models_without_free_tier": no_free,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.eval.cost")
    ap.add_argument("run_dir")
    ap.add_argument("--articles", type=int, default=228, help="Articles in the target runs (test set: 228)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--as-config", help="Price the run's calls as this config's models (for fake runs)")
    ap.add_argument("--prices", default="configs/prices.yaml")
    ap.add_argument("--fake-output-tokens", type=int, default=400)
    args = ap.parse_args(argv)

    prices = yaml.safe_load(Path(args.prices).read_text())["models"]
    as_cfg = load_run_config(args.as_config) if args.as_config else None
    r = estimate(Path(args.run_dir), prices, args.articles, args.repeats, as_cfg, args.fake_output_tokens)

    print(f"Estimate: {r['target_articles']} articles x {r['repeats']} repeats, priced as {r['priced_as']}")
    print(f"{'role':12} {'calls/art':>9} {'in tok':>8} {'out tok':>8} {'$/article':>10} {'projected $':>12}  models")
    for row in r["by_role"]:
        print(f"{row['role']:12} {row['calls_per_article']:9} {row['input_tokens_per_call']:8} "
              f"{row['output_tokens_per_call']:8} {row['usd_per_article']:10.5f} {row['usd_projected']:12.2f}  {row['models']}")
    print(f"{'TOTAL':12} {'':9} {'':8} {'':8} {'':10} {r['total_usd']:12.2f}")
    if r["estimated_from_fake_model"]:
        print(f"\nRough: token counts come from the fake model (outputs assumed {args.fake_output_tokens} tokens/call). "
              "Re-estimate from a real calibration run.")
    if r["models_without_free_tier"]:
        print(f"Needs a paid/licensed key (no free tier): {', '.join(r['models_without_free_tier'])}")
    out = Path(args.run_dir) / "cost_estimate.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
