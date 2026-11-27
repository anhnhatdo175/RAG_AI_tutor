"""LightGCN for task ranking (card P2-06, review B4), same protocol and metrics as ``lmsrag.ranking.tasks``.

LightGCN (He et al. 2020): learner and item embeddings (64 dims) propagated over the normalised learner-item graph
for 3 layers, layer mean as the final embedding, BPR loss with one uniform negative per positive, Adam
(lr 1e-3, L2 1e-4 on the batch embeddings), batch 4,096.

* ``val`` phase: trained on ``history``, NDCG@10 on the ``validation`` targets every 5 epochs; the number of epochs
  (at most ``--max-epochs``, stop after 3 checks without gain) is the only thing chosen here.
* ``test`` phase: retrained on ``history + validation`` for that many epochs, 5 seeds, scored once on ``test``.

Candidates, targets, learner sample (``--sample-users`` with ``data_seed``) and metrics are those of ``tasks.py``, so
the row can be appended to ``b4_task_ranking_<dataset>.csv``. Needs PyTorch; a GPU (Kaggle) is used when present.

    python -m lmsrag.ranking.lightgcn --dataset assist2009
    python -m lmsrag.ranking.lightgcn --dataset junyi2015 --sample-users 4000

Output ``reports/tables/b4_lightgcn_<dataset>.csv`` (same columns as ``b4_task_ranking``) and
``b4_lightgcn_choices_<dataset>.json``.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from ..datasets.build import REPO, load_config
from . import tasks as T


def norm_adj(Xall: sparse.csr_matrix):
    import torch
    nu, ni = Xall.shape
    A = sparse.bmat([[None, Xall], [Xall.T, None]], format="coo")
    deg = np.asarray(A.sum(1)).ravel()
    d = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    A = sparse.diags(d) @ A @ sparse.diags(d)
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data, dtype=torch.float32), (nu + ni, nu + ni)).coalesce()


class LightGCN:
    def __init__(self, Xall: sparse.csr_matrix, seed: int, dim: int = 64, layers: int = 3, lr: float = 1e-3,
                 reg: float = 1e-4, batch: int = 4096, device: str | None = None):
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nu, self.ni = Xall.shape
        self.A = norm_adj(Xall).to(self.dev)
        self.E = torch.nn.Parameter(torch.empty(self.nu + self.ni, dim, device=self.dev))
        torch.nn.init.normal_(self.E, std=0.1)
        self.opt = torch.optim.Adam([self.E], lr=lr)
        self.layers, self.reg, self.batch = layers, reg, batch
        coo = Xall.tocoo()
        self.pu, self.pi = coo.row, coo.col
        self.Xall = Xall

    def propagate(self):
        h, out = self.E, [self.E]
        for _ in range(self.layers):
            h = self.torch.sparse.mm(self.A, h)
            out.append(h)
        return self.torch.stack(out).mean(0)

    def epoch(self) -> float:
        torch = self.torch
        order = self.rng.permutation(len(self.pu))
        total = 0.0
        for a in range(0, len(order), self.batch):
            b = order[a:a + self.batch]
            u, i = self.pu[b], self.pi[b]
            j = self.rng.integers(0, self.ni, size=len(b))
            emb = self.propagate()
            tu = torch.as_tensor(u, device=self.dev)
            ti = torch.as_tensor(i + self.nu, device=self.dev)
            tj = torch.as_tensor(j + self.nu, device=self.dev)
            eu, ei, ej = emb[tu], emb[ti], emb[tj]
            loss = -torch.nn.functional.logsigmoid((eu * ei).sum(1) - (eu * ej).sum(1)).mean()
            loss = loss + self.reg * (self.E[tu].pow(2).sum() + self.E[ti].pow(2).sum()
                                      + self.E[tj].pow(2).sum()) / (2 * len(b))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item() * len(b)
        return total / len(order)

    def rank(self, rows: np.ndarray, seen: sparse.csr_matrix, k: int = T.K) -> np.ndarray:
        """Top-k unseen items for the given rows of ``Xall`` (one row per evaluated learner)."""
        torch = self.torch
        with torch.no_grad():
            emb = self.propagate()
            U, I = emb[:self.nu], emb[self.nu:]
            out = []
            for a in range(0, len(rows), 1024):
                r = rows[a:a + 1024]
                s = U[torch.as_tensor(r, device=self.dev)] @ I.T
                m = torch.as_tensor(seen[a:a + 1024].toarray(), device=self.dev, dtype=torch.bool)
                s = s.masked_fill(m, float("-inf"))
                out.append(torch.topk(s, k, dim=1).indices.cpu().numpy())
        return np.vstack(out)


def ranked_for(model: LightGCN, ph: T.Phase) -> dict:
    allu = np.unique(ph.train["user_id"])
    known = set(allu)
    users = [u for u in ph.targets if u in known]
    rows = np.searchsorted(allu, users)
    seen = ph.X[[ph.uidx[u] for u in users]]
    top = model.rank(rows, seen)
    return dict(zip(users, top))


def run(df: pd.DataFrame, cfg_mastery: dict, seed: int, max_epochs: int, check: int = 5, patience: int = 3,
        seeds=T.SEEDS):
    df = df.copy()
    df["source_skill"] = df["source_skill"].astype(str)
    item_index = pd.Index(np.sort(df["item_id"].unique()))
    item_skill_name = df.drop_duplicates("item_id").set_index("item_id")["source_skill"].reindex(item_index).to_numpy()
    skills = pd.Index(np.sort(df["source_skill"].unique()))
    item_skill = skills.get_indexer(item_skill_name)
    t0 = time.time()

    def make(train_splits, target_split):
        ph = T.Phase(df, train_splits, target_split, item_index, item_skill, None, None)
        ph.item_skill_name = item_skill_name
        return ph

    zero = lambda u, it: np.full(len(it), T.TARGET)  # noqa: E731  (zpd_gap is not used for the choice)
    ph_v = make(("history",), "validation")
    m = LightGCN(ph_v.Xall, seed)
    curve, best, bad = {}, (-1.0, 0), 0
    for ep in range(1, max_epochs + 1):
        loss = m.epoch()
        if ep % check:
            continue
        nd = T.metrics_for(ranked_for(m, ph_v), ph_v, zero)["ndcg@10"]
        curve[ep] = round(float(nd), 5)
        print(f"val epoch {ep}: loss {loss:.4f} ndcg@10 {nd:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if nd > best[0]:
            best, bad = (nd, ep), 0
        else:
            bad += 1
            if bad >= patience:
                break
    epochs = best[1]
    ph_t = make(("history", "validation"), "test")
    sc = T.Scorer(df, ph_t, item_index, skills, cfg_mastery, seed)
    rows = []
    for s in seeds:
        mt = LightGCN(ph_t.Xall, s)
        for _ in range(epochs):
            mt.epoch()
        r = T.metrics_for(ranked_for(mt, ph_t), ph_t, lambda u, it: sc.phat(ph_t.uidx[u], it))
        rows.append({"setting": "all", "model": "lightgcn", "seed": s, **r})
        print(f"test seed {s}: ndcg@10 {r['ndcg@10']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(rows)
    cols = ["recall@10", "ndcg@10", "mrr@10", "zpd_gap@10", "skill_cover@10", "new_item_recall@10", "n_learners"]
    agg = res.groupby(["setting", "model"]).agg(**{c: (c, "mean") for c in cols}, ndcg_sd=("ndcg@10", "std"),
                                                n_seeds=("seed", "nunique")).reset_index()
    return agg, res, {"epochs": epochs, "val_ndcg_by_epoch": curve, "device": m.dev}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--sample-users", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    df = pd.read_parquet(out_root / f"interactions_{args.dataset}.parquet")
    if args.sample_users:
        from ..datasets.splits import sample_users
        df = sample_users(df, args.sample_users, seed)
    agg, res, choice = run(df, T.mastery_cfg(tables, args.dataset), seed, args.max_epochs)
    agg.round(5).to_csv(tables / f"b4_lightgcn_{args.dataset}.csv", index=False)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(out_root / "predictions" / f"b4_lightgcn_seeds_{args.dataset}.csv", index=False)
    (tables / f"b4_lightgcn_choices_{args.dataset}.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(agg.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
