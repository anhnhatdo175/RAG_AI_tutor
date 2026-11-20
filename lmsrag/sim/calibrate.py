"""Check the simulator against Junyi (card P1-08) and write its report tables.

    python -m lmsrag.sim.calibrate --config configs/sim.yaml

- ``reports/tables/a6_sim_calibration.csv``: per skill, the Junyi history accuracy, mu_k, the accuracy the model gives
  (quadrature) and the first-answer accuracy of ``check.n_learners`` simulated learners (one random item of the skill
  each, through ``SimEnv.step``). Done when every |sim_rate - junyi_rate| < 0.05.
- ``reports/tables/a6_sim_learning.csv``: per learning rule, turns until theta_{u,k} has risen by one unit when the
  items are drawn at random from one skill (each learner one random skill), and the same at a constant P = 0.7.
- ``reports/tables/a6_sim_items.csv``: items kept and dropped by source and difficulty tier.

    python -m lmsrag.sim.calibrate --config configs/sim.yaml --tune-eta

chooses the eta range (``tune_eta``) on ``world_seed + 1`` and writes ``reports/tables/a6_sim_eta_tuning.csv``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from .env import SimEnv
from .model import RULES, World, learning_gain, load_world

TOL = 0.05


def initial_accuracy(world: World, cfg: dict, n_learners: int, seed: int) -> pd.DataFrame:
    env = SimEnv.from_config(world, cfg, "flat", n_learners)
    env.reset(seed)
    rng = np.random.default_rng([seed, 1])
    by_skill = world.items.groupby("skill_tag")["item_id"].apply(list)
    hits = {}
    for k, ids in by_skill.items():
        picks = rng.integers(0, len(ids), size=n_learners)
        # learning only moves theta of the skill answered, so each first answer per skill is a pre-learning answer
        hits[k] = np.mean([env.step(u, ids[j]) for u, j in enumerate(picks)])
    out = world.skills.copy()
    out["sim_rate"] = pd.Series(hits)
    out["sim_n"] = n_learners
    out["dev"] = out["sim_rate"] - out["junyi_rate"]
    out["model_dev"] = out["model_rate"] - out["junyi_rate"]
    return out.reset_index().rename(columns={"index": "skill_tag"})


def turns_to_unit(world: World, cfg: dict, rule: str, n_learners: int, seed: int, max_turns: int) -> np.ndarray:
    """Per learner: turns until theta_{u,k} has risen by 1 on one random skill k, items drawn at random from k
    (the ``random`` policy of P1-09 within a skill); ``inf`` when not reached in ``max_turns``. The same seed gives
    the same learners, skills and item draws whatever ``cfg["eta"]`` is, so runs with two eta ranges are paired."""
    env = SimEnv.from_config(world, cfg, rule, n_learners)
    env.reset(seed)
    rng = np.random.default_rng([seed, 2])
    by_skill = world.items.groupby("skill_tag")["item_id"].apply(list)
    skills = list(by_skill.index)
    turns = np.full(n_learners, np.inf)
    for u in range(n_learners):
        k = skills[rng.integers(len(skills))]
        ids, start = by_skill[k], env.true_theta(u, k)
        draws = rng.integers(len(ids), size=max_turns)
        for t in range(max_turns):
            env.step(u, ids[draws[t]])
            if env.true_theta(u, k) - start >= 1.0:
                turns[u] = t + 1
                break
    return turns


def turns_per_unit(world: World, cfg: dict, rule: str, n_learners: int, seed: int, max_turns: int,
                   turns: np.ndarray | None = None) -> dict:
    if turns is None:
        turns = turns_to_unit(world, cfg, rule, n_learners, seed, max_turns)
    reached = np.isfinite(turns)
    lo, hi = cfg["eta"]
    z = cfg.get("zpd", {"center": 0.7, "width": 0.15})
    inv_eta = np.log(hi / lo) / (hi - lo)  # E[1 / eta] for eta ~ U(lo, hi)
    g07 = float(learning_gain(0.7, rule, z["center"], z["width"]))
    return {"rule": rule, "policy": "random_in_skill", "n_learners": n_learners, "eta_low": lo, "eta_high": hi,
            "median_turns": float(np.median(turns)),  # unreached count as above the median
            "mean_turns_reached": float(turns[reached].mean()) if reached.any() else np.nan,
            "share_reached": float(reached.mean()), "max_turns": max_turns,
            "turns_at_p07_mean_learner": 1.0 / (np.mean([lo, hi]) * g07),
            "turns_at_p07_mean_over_learners": inv_eta / g07}


def tune_eta(world: World, cfg: dict, target: float, n_learners: int, seed: int, max_turns: int,
             iters: int = 12) -> tuple[tuple[float, float], pd.DataFrame]:
    """Scale the eta range (keeping high / low) until the median turns to +1 theta, pooled over the three learning
    rules with random items in a skill, equals ``target``. Turns fall as eta rises; the median of whole turns is a
    step function, so the search is a bisection on log(scale) and returns the smallest scale whose median is at most
    ``target``. Runs on ``world_seed + 1`` (the tuning world of P1-09), so the final world is not used to choose eta."""
    lo0, hi0 = (float(x) for x in cfg["eta"])
    log = []

    def pooled(scale):
        c = {**cfg, "eta": [lo0 * scale, hi0 * scale]}
        t = np.concatenate([turns_to_unit(world, c, r, n_learners, seed, max_turns) for r in RULES])
        m = float(np.median(t))
        log.append({"scale": scale, "eta_low": lo0 * scale, "eta_high": hi0 * scale, "pooled_median_turns": m,
                    "share_reached": float(np.isfinite(t).mean())})
        return m

    a, b = 0.25, 4.0  # scale bracket: median(a) > target >= median(b)
    if not (pooled(a) > target >= pooled(b)):
        raise ValueError(f"target {target} turns is outside the bracket {log}")
    for _ in range(iters):
        mid = float(np.sqrt(a * b))
        if pooled(mid) > target:
            a = mid
        else:
            b = mid
    scale = b
    return (lo0 * scale, hi0 * scale), pd.DataFrame(log)


def item_summary(bank: pd.DataFrame, world: World) -> pd.DataFrame:
    kept = set(world.items["item_id"])
    b = bank.assign(kept=bank["chunk_id"].isin(kept))
    t = b.groupby(["source", "difficulty_tier"]).agg(items=("chunk_id", "size"), kept=("kept", "sum")).reset_index()
    bq = world.items.groupby(["source", "difficulty_tier"])["b"].quantile([0.1, 0.5, 0.9]).unstack()
    bq.columns = ["b_p10", "b_p50", "b_p90"]
    return t.merge(bq.reset_index(), on=["source", "difficulty_tier"], how="left")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "sim.yaml")
    ap.add_argument("--tables", type=Path, default=REPO / "reports" / "tables")
    ap.add_argument("--tune-eta", action="store_true",
                    help="search the eta range for eta_target_turns on world_seed + 1, write a6_sim_eta_tuning.csv "
                         "and stop; copy the result into configs/sim.yaml by hand")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    t0 = time.time()
    chk = cfg.get("check", {})
    n, seed, max_turns = int(chk.get("n_learners", 10000)), int(chk.get("seed", 0)), int(chk.get("max_turns", 400))
    args.tables.mkdir(parents=True, exist_ok=True)
    if args.tune_eta:
        tune = cfg["eta_tuning"]
        tw = load_world(cfg, REPO, seed=int(cfg["world_seed"]) + 1)
        start = {**cfg, "eta": tune["start"]}
        (lo, hi), log = tune_eta(tw, start, float(tune["target_turns"]), int(cfg["n_learners"]), seed, max_turns)
        log.to_csv(args.tables / "a6_sim_eta_tuning.csv", index=False, float_format="%.5g")
        print(log.to_string(index=False))
        print(f"eta: [{lo:.4f}, {hi:.4f}]  (done in {time.time() - t0:.0f} s)")
        return 0
    world = load_world(cfg, REPO)

    cal = initial_accuracy(world, cfg, n, seed)
    cols = ["skill_tag", "n_items", "junyi_n", "junyi_rate", "mu", "model_rate", "sim_rate", "sim_n", "dev", "model_dev"]
    cal[cols].sort_values("junyi_n", ascending=False).to_csv(args.tables / "a6_sim_calibration.csv", index=False,
                                                            float_format="%.6g")
    learn = pd.DataFrame([turns_per_unit(world, cfg, r, int(cfg["n_learners"]), seed, max_turns) for r in RULES])
    learn.to_csv(args.tables / "a6_sim_learning.csv", index=False, float_format="%.4g")
    bank = pd.read_json(REPO / cfg["items"], lines=True)
    items = item_summary(bank, world)
    items.to_csv(args.tables / "a6_sim_items.csv", index=False, float_format="%.4g")

    worst = cal.loc[cal["dev"].abs().idxmax()]
    print(f"skills {len(world.skills)}, items {len(world.items)} of {len(bank)}; "
          f"max |sim - junyi| {abs(worst['dev']):.4f} ({worst['skill_tag']}), max |model - junyi| "
          f"{cal['model_dev'].abs().max():.4f}; skills over {TOL}: {(cal['dev'].abs() >= TOL).sum()}")
    print(learn.to_string(index=False))
    print(items.to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")
    return 0 if (cal["dev"].abs() < TOL).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
