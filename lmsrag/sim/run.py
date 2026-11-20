"""Closed-loop experiment on simulated learners (card P1-09, plan item C5). All numbers it writes are simulation results.

    python -m lmsrag.sim.run --config configs/sim.yaml --world tune    # world_seed + 1: choose proposed's parameters
    python -m lmsrag.sim.run --config configs/sim.yaml --world final   # world_seed, once, after Mốc B

Design: ``n_learners`` x ``turns`` x seeds x the three learning rules. Each learner practises one skill, drawn with
the seed by ``closed_loop.skill_assignment`` (``practised_skills``: uniform over the skills the learner has not
mastered yet), the same for every policy and rule; for a given seed, every policy meets the same learners
(``SimEnv.reset(seed)``), so policy differences are paired by learner. ``closed_loop.references`` (``oracle``) are
reported next to the policies as a ceiling; they read true parameters and are not policies.

Policies (``closed_loop.policies``): ``random``, ``fixed`` (always level 3), ``most_popular`` (most Junyi history
responses), ``proposed`` (``lmsrag.sim.policies.ProposedPolicy`` of P2-08). A policy gets ``observe(u)`` and the public
catalog rows of the learner's skill, nothing else. ``ProposedPolicy`` is built with ``ProposedPolicy.from_config(cfg,
catalog)`` when it has that classmethod, else ``ProposedPolicy()``.

Metrics per learner:

- ``zpd_hit_rate``: share of turns whose item had ``true_p`` in [0.6, 0.8] before the answer
- ``mastery_gain``: theta after the last turn minus theta before the first, on the practised skill
- ``turns_to_mastery``: first turn after which sigmoid(a_m (theta - b_m)) >= 0.95 on the median-b item m of the skill;
  ``turns`` when not reached (``not_mastered`` = 1), 0 when mastered before the first turn (``mastered_at_start``)

Tables (suffix ``_tune`` for ``--world tune``): ``reports/tables/c5_closed_loop.csv`` (mean over seeds, sd over seeds,
95% bootstrap interval by learner), ``c5_closed_loop_vs_<compare_to>.csv`` (paired differences), ``c5_mastery_curves.csv``
and ``reports/figures/c5_mastery_curves.png``.
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from .baselines import FixedLevelPolicy, MostPopularPolicy, RandomPolicy, junyi_popularity
from .env import SimEnv
from .model import RULES, World, load_world

METRICS = ["zpd_hit_rate", "mastery_gain", "turns_to_mastery", "not_mastered", "mastered_at_start"]
LABEL = "Kết quả mô phỏng"


def load_proposed(spec: str, cfg: dict, catalog: pd.DataFrame):
    mod, _, name = spec.partition(":")
    cls = getattr(importlib.import_module(mod), name)
    return cls.from_config(cfg, catalog) if hasattr(cls, "from_config") else cls()


def make_policies(names, cfg: dict, catalog: pd.DataFrame, popularity: dict | None, required: bool) -> dict:
    cl = cfg["closed_loop"]
    out = {}
    for name in names:
        if name == "random":
            out[name] = RandomPolicy()
        elif name == "fixed":
            out[name] = FixedLevelPolicy(int(cl.get("fixed_level", 3)))
        elif name == "most_popular":
            if popularity is None:
                raise ValueError("most_popular needs Junyi popularity counts")
            out[name] = MostPopularPolicy(popularity)
        elif name == "oracle":
            lo, hi = cl.get("zpd_band", (0.6, 0.8))
            out[name] = OracleReference((float(lo) + float(hi)) / 2)
        elif name == "proposed":
            try:
                out[name] = load_proposed(cl["proposed"], cfg, catalog)
            except (ImportError, AttributeError) as e:
                if required:
                    raise SystemExit(f"proposed policy {cl['proposed']!r} is required here: {e}")
                print(f"notice: proposed policy {cl['proposed']!r} not available ({e}); running without it")
        else:
            raise ValueError(f"unknown policy {name!r}")
    return out


ASSIGNMENTS = ("uniform", "junyi", "uniform_unmastered", "junyi_unmastered")


def assign_skills(skills: list[str], n: int, seed: int) -> list[str]:
    """Uniform over the skills (``skill_assignment: uniform``)."""
    rng = np.random.default_rng([int(seed), 3])
    return [skills[i] for i in rng.integers(0, len(skills), size=n)]


def practised_skills(env: SimEnv, world: World, seed: int, mode: str, threshold: float) -> tuple[list[str], int]:
    """The skill each learner practises, drawn with ``seed``; the same for every policy and learning rule (the
    learners of a seed do not depend on the rule). ``uniform``: all skills alike; ``junyi``: in proportion to the
    skill's Junyi history responses; ``*_unmastered``: only skills the learner has not mastered before the first turn
    (sigmoid(a_m (theta - b_m)) < threshold on the median item), the way a tutor gives practice. Uses true parameters,
    but only to set up the experiment: a policy still sees nothing but observe(u). Returns the skills and the number
    of learners with no unmastered skill (they fall back to all skills)."""
    if mode not in ASSIGNMENTS:
        raise ValueError(f"unknown skill_assignment {mode!r}, expected one of {ASSIGNMENTS}")
    skills = env.skills
    if mode == "uniform":
        return assign_skills(skills, env.n_learners, seed), 0
    w = world.skills.loc[skills, "junyi_n"].to_numpy(dtype="float64") if mode.startswith("junyi")         else np.ones(len(skills))
    env.reset(seed)
    elig = np.ones((env.n_learners, len(skills)), dtype=bool)
    if mode.endswith("_unmastered"):
        med = [env.median_item(k) for k in skills]
        elig = np.array([[env.true_mastery(u, m) < threshold for m in med] for u in range(env.n_learners)])
    rng = np.random.default_rng([int(seed), 3])
    out, fallback = [], 0
    for u in range(env.n_learners):
        wu = w * elig[u]
        if wu.sum() == 0:
            wu, fallback = w, fallback + 1
        out.append(skills[int(rng.choice(len(skills), p=wu / wu.sum()))])
    return out, fallback


class OracleReference:
    """Reference ceiling, not a policy: reads the true P of every candidate and gives the one nearest ``target``
    (repeats allowed). Shown next to the policies to read how much of the reachable ZPD hit rate they get."""
    name = "oracle"
    reference = True

    def __init__(self, target: float = 0.7):
        self.target = float(target)

    def choose_env(self, env: SimEnv, u: int, candidates: pd.DataFrame) -> str:
        ids = candidates["item_id"].to_numpy()
        return str(ids[int(np.argmin(np.abs(env.true_p_many(u, ids) - self.target)))])


def run_policy(env: SimEnv, policy, seed: int, practised: list[str], candidates: dict[str, pd.DataFrame],
               turns: int, band: tuple[float, float], threshold: float) -> dict[str, np.ndarray]:
    """One policy, one seed, one learning rule. Returns per-learner metrics and the mean gain curve (turns + 1)."""
    env.reset(seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    n = env.n_learners
    hit, gain, ttm = np.zeros(n), np.zeros(n), np.full(n, float(turns))
    not_m, at_start = np.ones(n), np.zeros(n)
    curve = np.zeros(turns + 1)
    allowed = {k: set(c["item_id"]) for k, c in candidates.items()}
    median = {k: env.median_item(k) for k in candidates}
    lo, hi = band
    for u in range(n):
        k = practised[u]
        cands, m = candidates[k], median[k]
        start = env.true_theta(u, k)
        mastered = env.true_mastery(u, m) >= threshold
        if mastered:
            ttm[u], not_m[u], at_start[u] = 0.0, 0.0, 1.0
        hits = 0
        for t in range(turns):
            item = (policy.choose_env(env, u, cands) if getattr(policy, "reference", False)
                    else policy.choose(env.observe(u), cands))
            if item not in allowed[k]:
                raise ValueError(f"{type(policy).__name__} chose {item!r}, not a candidate of skill {k!r}")
            p = env.true_p(u, item)
            hits += lo <= p <= hi
            env.step(u, item)
            curve[t + 1] += env.true_theta(u, k) - start
            if not mastered and env.true_mastery(u, m) >= threshold:
                mastered, ttm[u], not_m[u] = True, float(t + 1), 0.0
        hit[u] = hits / turns
        gain[u] = env.true_theta(u, k) - start
    return {"zpd_hit_rate": hit, "mastery_gain": gain, "turns_to_mastery": ttm, "not_mastered": not_m,
            "mastered_at_start": at_start, "curve": curve / n}


def bootstrap_index(n_seeds: int, n_learners: int, n_resamples: int, seed: int) -> np.ndarray:
    """(seeds, resamples, learners) learner indices, shared by all policies so differences stay paired."""
    rng = np.random.default_rng([int(seed), 9])
    return rng.integers(0, n_learners, size=(n_seeds, n_resamples, n_learners))


def summarise(values: np.ndarray, idx: np.ndarray) -> dict:
    """values: (seeds, learners). Mean over seeds of the per-seed means, sd of the per-seed means, and the 2.5/97.5
    percentiles of that mean when learners are resampled within each seed."""
    per_seed = values.mean(axis=1)
    boot = np.stack([values[s][idx[s]].mean(axis=1) for s in range(values.shape[0])]).mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(per_seed.mean()), "sd_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
            "ci_low": float(lo), "ci_high": float(hi), "boot": boot}


def run(world: World, cfg: dict, policies: dict, popularity_note: str = "") -> tuple[pd.DataFrame, pd.DataFrame,
                                                                                         pd.DataFrame]:
    cl = cfg["closed_loop"]
    n, turns, seeds = int(cl["n_learners"]), int(cl["turns"]), [int(s) for s in cl["seeds"]]
    band, thr = tuple(cl.get("zpd_band", (0.6, 0.8))), float(cl.get("mastery_threshold", 0.95))
    catalog = world.catalog()
    candidates = {k: g.reset_index(drop=True) for k, g in catalog.groupby("skill_tag")}
    idx = bootstrap_index(len(seeds), n, int(cl.get("n_resamples", 1000)), int(cfg["world_seed"]))
    rows, diffs, curves = [], [], []
    ref = cl.get("compare_to", "fixed")
    mode = cl.get("skill_assignment", "uniform")
    practised: dict[int, list[str]] = {}
    for rule in cfg.get("learning_rules", RULES):
        env = SimEnv.from_config(world, cfg, rule, n)
        if not practised:  # learners do not depend on the rule, so one assignment per seed serves every rule
            for s in seeds:
                practised[s], fallback = practised_skills(env, world, s, mode, thr)
                if fallback:
                    print(f"  seed {s}: {fallback} learners have no unmastered skill; they draw from all skills")
        res = {}
        for name, pol in policies.items():
            t0 = time.time()
            per = [run_policy(env, pol, s, practised[s], candidates, turns, band, thr) for s in seeds]
            res[name] = {m: np.stack([r[m] for r in per]) for m in METRICS}
            c = np.stack([r["curve"] for r in per])
            curves += [{"rule": rule, "policy": name, "turn": t, "mean_gain": float(c[:, t].mean()),
                        "sd_seeds": float(c[:, t].std(ddof=1)) if len(seeds) > 1 else np.nan} for t in range(turns + 1)]
            print(f"  {rule:4s} {name:12s} {time.time() - t0:5.1f} s")
        for name in policies:
            for m in METRICS:
                s = summarise(res[name][m], idx)
                rows.append({"label": LABEL, "rule": rule, "policy": name,
                             "kind": "reference" if getattr(policies[name], "reference", False) else "policy",
                             "skill_assignment": mode, "metric": m, "mean": s["mean"],
                             "sd_seeds": s["sd_seeds"], "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                             "n_learners": n, "n_seeds": len(seeds), "turns": turns, "world_seed": world.seed})
        if ref in res:
            for name in policies:
                if name == ref:
                    continue
                for m in METRICS:
                    s = summarise(res[name][m] - res[ref][m], idx)
                    b = s["boot"]
                    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
                    diffs.append({"label": LABEL, "rule": rule, "policy": name, "vs": ref, "metric": m,
                                  "diff": s["mean"], "sd_seeds": s["sd_seeds"], "ci_low": s["ci_low"],
                                  "ci_high": s["ci_high"], "p_boot": p, "n_learners": n, "n_seeds": len(seeds),
                                  "world_seed": world.seed})
    return pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(curves)


# ---------------------------------------------------------------- figure
COLORS = {"random": "#2a78d6", "fixed": "#eb6834", "most_popular": "#1baf7a", "proposed": "#eda100"}  # fixed order
NAMES = {"random": "random", "fixed": "fixed (mức 3)", "most_popular": "most_popular", "proposed": "proposed",
         "oracle": "oracle (trần, dùng P thật)"}
REF_STYLE = {"oracle": {"color": "#52514e", "ls": (0, (4, 3))}}  # a reference is not a series: muted ink, dashed
RULE_TITLES = {"flat": "flat: G = 1", "info": "info: G = 4P(1 − P)", "zpd": "zpd: G đỉnh tại P = 0,7"}


def plot_curves(curves: pd.DataFrame, path: Path, world_seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = list(dict.fromkeys(curves["rule"]))
    pols = [p for p in list(COLORS) + list(REF_STYLE) if p in set(curves["policy"])]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    fig, axes = plt.subplots(1, len(rules), figsize=(4.2 * len(rules) + 1.2, 4.0), sharey=True, dpi=150)
    axes = np.atleast_1d(axes)
    ymax = curves["mean_gain"].max()
    for ax, rule in zip(axes, rules):
        c = curves[curves["rule"] == rule]
        ends = []
        for p in pols:
            d = c[c["policy"] == p]
            style = REF_STYLE.get(p, {"color": COLORS.get(p), "ls": "-"})
            ax.plot(d["turn"], d["mean_gain"], color=style["color"], ls=style["ls"], lw=2 if p in COLORS else 1.5,
                    label=NAMES[p], solid_capstyle="round")
            ends.append((float(d["mean_gain"].iloc[-1]), p, int(d["turn"].iloc[-1])))
        # direct labels at the line ends, nudged apart so they never overlap
        ends.sort()
        placed = []
        gap = ymax * 0.07
        for y, p, x in ends:
            y_lab = max(y, placed[-1] + gap) if placed else y
            placed.append(y_lab)
            ax.annotate(p, xy=(x, y), xytext=(x + 1.5, y_lab), fontsize=8, color=ink, va="center",
                        arrowprops=None)
        ax.set_title(RULE_TITLES.get(rule, rule), fontsize=10, color=ink, loc="left")
        ax.set_xlim(0, c["turn"].max() + 14)
        ax.set_xticks(range(0, int(c["turn"].max()) + 1, 10))
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlabel("Lượt", fontsize=9, color=muted)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("Tăng θ trung bình trên kỹ năng được luyện", fontsize=9, color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(pols), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{LABEL}: tăng θ theo lượt, theo chính sách chọn bài (world {world_seed})", fontsize=11, color=ink,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "sim.yaml")
    ap.add_argument("--world", choices=["tune", "final"], required=True)
    ap.add_argument("--policies", help="comma-separated subset of closed_loop.policies")
    ap.add_argument("--n-learners", type=int, help="override closed_loop.n_learners (quick runs only)")
    ap.add_argument("--tables", type=Path, default=REPO / "reports" / "tables")
    ap.add_argument("--figures", type=Path, default=REPO / "reports" / "figures")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cl = cfg["closed_loop"]
    if args.n_learners:
        cl["n_learners"] = args.n_learners
    names = args.policies.split(",") if args.policies else list(cl["policies"]) + list(cl.get("references", []))
    final = args.world == "final"
    if final and (args.policies or args.n_learners):
        raise SystemExit("--world final runs the configured design as is: no --policies or --n-learners")
    t0 = time.time()
    seed = int(cfg["world_seed"]) + (0 if final else 1)
    world = load_world(cfg, REPO, seed=seed)
    cfg = {**cfg, "world_seed": seed}
    catalog = world.catalog()
    popularity = None
    if "most_popular" in names:
        junyi = pd.read_parquet(REPO / cfg["junyi"], columns=["item_id", "split"])
        types = pd.read_csv(REPO / "lmsrag/data/khan_v2/khan_problem_types.csv", dtype=str)
        popularity = junyi_popularity(catalog, junyi, types)
    policies = make_policies(names, cfg, catalog, popularity, required=final)
    print(f"world {seed}: {len(world.items)} items, {len(world.skills)} skills; policies {list(policies)}")
    table, diffs, curves = run(world, cfg, policies)
    suffix = "" if final else "_tune"
    args.tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.tables / f"c5_closed_loop{suffix}.csv", index=False, float_format="%.5g")
    if len(diffs):
        diffs.to_csv(args.tables / f"c5_closed_loop_vs_{cl.get('compare_to', 'fixed')}{suffix}.csv", index=False,
                     float_format="%.5g")
    curves.to_csv(args.tables / f"c5_mastery_curves{suffix}.csv", index=False, float_format="%.5g")
    plot_curves(curves, args.figures / f"c5_mastery_curves{suffix}.png", seed)
    show = table[table["metric"].isin(["zpd_hit_rate", "mastery_gain", "turns_to_mastery", "not_mastered"])]
    print(show.pivot_table(index=["rule", "policy"], columns="metric", values="mean").round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
