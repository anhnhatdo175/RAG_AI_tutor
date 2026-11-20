"""Choose the skill assignment of the closed-loop experiment (card P1-09) on the tuning world, before any test run.

    python -m lmsrag.sim.design_check --config configs/sim.yaml [--seeds 0,1]

Runs ``random``, ``fixed``, ``most_popular`` and the ``oracle`` reference for each ``skill_assignment`` in
``ASSIGNMENTS`` on ``world_seed + 1``; ``proposed`` takes no part, so the choice cannot favour it. Decision rule, fixed
before the first run (``simulation.md``, P1-09):

1. validity: no learner practises a skill mastered before the first turn (``mastered_at_start`` = 0);
2. sensitivity: (oracle - fixed) ZPD hit rate divided by the half-width of its paired 95% interval, mean over the three
   learning rules; higher is better;
3. a tie (within 10% on rule 2) goes to the simpler assignment, in the order of ``ASSIGNMENTS``.

Writes ``reports/tables/c5_design_check_tune.csv``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ..datasets.build import REPO, load_config
from .baselines import junyi_popularity
from .model import load_world
from .run import ASSIGNMENTS, make_policies, run

NAMES = ["random", "fixed", "most_popular", "oracle"]


def check(world, cfg: dict, popularity: dict, seeds: list[int], modes=ASSIGNMENTS) -> pd.DataFrame:
    rows = []
    for mode in modes:
        t0 = time.time()
        c = {**cfg, "closed_loop": {**cfg["closed_loop"], "skill_assignment": mode, "seeds": seeds}}
        pols = make_policies(NAMES, c, world.catalog(), popularity, required=False)
        table, diffs, _ = run(world, c, pols)
        for rule in table["rule"].unique():
            t = table[table["rule"] == rule].pivot_table(index="policy", columns="metric", values="mean")
            d = diffs[(diffs["rule"] == rule) & (diffs["policy"] == "oracle") & (diffs["metric"] == "zpd_hit_rate")]
            half = float((d["ci_high"] - d["ci_low"]).iloc[0] / 2)
            rows.append({"skill_assignment": mode, "rule": rule,
                         "mastered_at_start": float(t.loc["fixed", "mastered_at_start"]),
                         **{f"hit_{p}": float(t.loc[p, "zpd_hit_rate"]) for p in NAMES},
                         **{f"gain_{p}": float(t.loc[p, "mastery_gain"]) for p in NAMES},
                         "not_mastered_fixed": float(t.loc["fixed", "not_mastered"]),
                         "headroom_hit": float(d["diff"].iloc[0]), "headroom_ci_half": half,
                         "sensitivity": float(d["diff"].iloc[0]) / half if half > 0 else float("inf"),
                         "n_seeds": len(seeds), "world_seed": world.seed})
        print(f"{mode}: {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)


def decide(res: pd.DataFrame, tie: float = 0.10) -> tuple[str, pd.DataFrame]:
    s = res.groupby("skill_assignment", sort=False).agg(mastered_at_start=("mastered_at_start", "max"),
                                                       sensitivity=("sensitivity", "mean"))
    s = s.reindex([m for m in ASSIGNMENTS if m in s.index])
    valid = s[s["mastered_at_start"] == 0] if (s["mastered_at_start"] == 0).any() else s
    best = valid["sensitivity"].max()
    pick = next(m for m in valid.index if valid.at[m, "sensitivity"] >= (1 - tie) * best)
    s["valid"] = s.index.isin(valid.index)
    s["chosen"] = s.index == pick
    return pick, s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "sim.yaml")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--tables", type=Path, default=REPO / "reports" / "tables")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg["world_seed"]) + 1
    world = load_world(cfg, REPO, seed=seed)
    cfg = {**cfg, "world_seed": seed}
    junyi = pd.read_parquet(REPO / cfg["junyi"], columns=["item_id", "split"])
    types = pd.read_csv(REPO / "lmsrag/data/khan_v2/khan_problem_types.csv", dtype=str)
    res = check(world, cfg, junyi_popularity(world.catalog(), junyi, types), [int(x) for x in args.seeds.split(",")])
    pick, summary = decide(res)
    args.tables.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.tables / "c5_design_check_tune.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(summary.round(3).to_string())
    print(f"chosen skill_assignment: {pick}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
