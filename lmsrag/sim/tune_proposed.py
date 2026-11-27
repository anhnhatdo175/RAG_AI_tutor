"""Choose ``closed_loop.proposed_params`` of P2-08 on the tune world only (world_seed + 1).

Rule written before the search: maximise the ZPD hit rate averaged over the three learning rules (the quantity the
policy targets); ties within 0.002 go to the larger mastery gain. A small design (``--n-learners``, one seed) is enough
to rank the grid; the chosen point is then run in full by ``lmsrag.sim.run --world tune``. Output:
``reports/tables/c5_proposed_tuning_tune.csv``. The final world is never touched here.

    python -m lmsrag.sim.tune_proposed
"""
from __future__ import annotations

import argparse
import itertools
import time

import pandas as pd

from ..datasets.build import REPO, load_config
from .model import load_world
from .policies import ProposedPolicy
from .run import run

GRID = {"scale": [0.25, 0.5, 1.0], "theta0": [0.0, 1.0, 2.0, 3.0], "k0": [0.5, 1.0], "k_min": [0.1, 0.3, 0.5]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--n-learners", type=int, default=200)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["world_seed"]) + 1
    world = load_world(cfg, REPO, seed=seed)
    cfg = {**cfg, "world_seed": seed,
           "closed_loop": {**cfg["closed_loop"], "n_learners": args.n_learners, "seeds": [0], "n_resamples": 1}}
    rows, t0 = [], time.time()
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        table, _, _ = run(world, cfg, {"proposed": ProposedPolicy(**params)})
        m = table[table["policy"] == "proposed"].groupby("metric")["mean"].mean()
        rows.append({**params, "zpd_hit_rate": m["zpd_hit_rate"], "mastery_gain": m["mastery_gain"]})
    out = pd.DataFrame(rows).sort_values(["zpd_hit_rate", "mastery_gain"], ascending=False)
    top = out.iloc[0]
    near = out[out["zpd_hit_rate"] >= top["zpd_hit_rate"] - 0.002].sort_values("mastery_gain", ascending=False)
    out["chosen"] = out.index == near.index[0]
    out.round(4).to_csv(REPO / "reports" / "tables" / "c5_proposed_tuning_tune.csv", index=False)
    print(out.head(8).round(4).to_string(index=False))
    print("chosen:", out[out["chosen"]].iloc[0][list(GRID)].to_dict(), f"- {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
