"""Deep knowledge tracing baselines DKT and SAKT (card P2-01, review B1/C1), same split and protocol as the others.

Architectures follow pykt-toolkit 0.0.38 (``pykt.models.dkt.DKT``, ``pykt.models.sakt.SAKT``, emb_type ``qid``):

* DKT (Piech et al. 2015): interaction embedding (skill, correct) -> LSTM (hidden = embedding = 64) -> dropout -> linear over skills -> sigmoid; the prediction for step t+1 is the output of step t at the
  skill of step t+1.
* SAKT (Pandey & Karypis 2019): interaction + position embeddings as keys/values, the next skill's embedding as
  query, causal multi-head attention blocks with FFN, window ``--seq-len``.

The pykt package itself is not imported (its ``qdkt`` module imports ``turtle``/tkinter, and its data pipeline makes
its own folds and truncations); the models are re-written here so they can use the project's split:

* the "concept" is the canonical skill (``source_skill``), as for BKT/PFA/AFM;
* training windows come from ``history`` rows only; the number of epochs is chosen by validation AUC (early
  stopping, patience 3); nothing else is tuned;
* prediction is prequential: the prediction for a validation or test answer uses every earlier answer of the learner
  (history, validation and earlier test rows) as input, without updating the weights;
* rows of one attempt (same item, same time: one answer tagged with several skills, 9% of ASSISTments 2009 rows)
  never see each other's answer: DKT reads the state before the attempt, SAKT masks the attempt's keys. Without
  this the duplicated answer leaks the label (DKT on 2009 test: AUC 0.794 with the leak).

    python -m lmsrag.mastery.deep_kt --dataset assist2009 --model dkt
    python -m lmsrag.mastery.deep_kt --dataset assist2009 --model sakt

Outputs ``<output_dir>/predictions/b1_deep_<model>_<dataset>.parquet`` (row, user_id, split, y, <model>; ``row`` as
in ``b1v_<dataset>.parquet``), ``reports/tables/b1_deep_<model>_<dataset>.csv`` (AUC, log loss, RMSE on validation and
test) and ``b1_deep_<model>_choices_<dataset>.json``. ``python -m lmsrag.stats.bootstrap`` picks the predictions up.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config


def sequences(df: pd.DataFrame):
    """Per learner, in time order: skill ids, answers, split codes (0 history, 1 validation, 2 test), row ids and
    attempt ids (rows of one attempt share an id: same item at the same time, i.e. one answer tagged with several
    skills, 9% of ASSISTments 2009 rows)."""
    d = df.copy()
    d["row"] = np.arange(len(d))
    d = d.sort_values(["user_id", "ts", "row"], kind="mergesort")
    codes, skills = pd.factorize(d["source_skill"].astype(str))
    d["c"] = codes
    d["s"] = d["split"].map({"history": 0, "validation": 1, "test": 2}).to_numpy()
    prev = d.shift()
    new = ~((d["user_id"] == prev["user_id"]) & (d["item_id"] == prev["item_id"]) & (d["ts"] == prev["ts"]))
    d["g"] = np.cumsum(new.to_numpy(dtype=bool))
    seqs = [(u, g["c"].to_numpy(), g["correct"].to_numpy(dtype=np.int64), g["s"].to_numpy(), g["row"].to_numpy(),
             g["g"].to_numpy()) for u, g in d.groupby("user_id", sort=True)]
    return seqs, len(skills)


def context_index(g: np.ndarray) -> np.ndarray:
    """For target t = 1..n-1 of a window: the last input position before t's attempt (-1 when there is none).

    Rows of the same attempt never see each other's answer, which would leak the label."""
    n = len(g)
    first = np.arange(n)
    for i in range(1, n):
        if g[i] == g[i - 1]:
            first[i] = first[i - 1]
    return first[1:] - 1


def build_model(kind: str, num_c: int, emb: int, seq_len: int, heads: int, dropout: float):
    import torch
    from torch import nn

    class DKT(nn.Module):
        def __init__(self):
            super().__init__()
            self.inter = nn.Embedding(num_c * 2, emb)
            self.lstm = nn.LSTM(emb, emb, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(emb, num_c)

        def forward(self, q, r, qn, ctx):
            h, _ = self.lstm(self.inter(q + num_c * r))
            y = torch.sigmoid(self.out(self.drop(h)))  # y[:, j] = state after input j, over every skill
            y = y.gather(1, ctx.clamp(min=0).unsqueeze(-1).expand(-1, -1, num_c))
            return y.gather(2, qn.unsqueeze(-1)).squeeze(-1)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(emb, heads, dropout=dropout, batch_first=True)
            self.d1, self.n1 = nn.Dropout(dropout), nn.LayerNorm(emb)
            self.ffn = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb, emb))
            self.d2, self.n2 = nn.Dropout(dropout), nn.LayerNorm(emb)

        def forward(self, q, kv, mask):
            a, _ = self.attn(q, kv, kv, attn_mask=mask)
            a = self.n1(q + self.d1(a))
            return self.n2(a + self.d2(self.ffn(a)))

    class SAKT(nn.Module):
        def __init__(self):
            super().__init__()
            self.inter = nn.Embedding(num_c * 2, emb)
            self.ex = nn.Embedding(num_c, emb)
            self.pos = nn.Embedding(seq_len, emb)
            self.blocks = nn.ModuleList([Block() for _ in range(2)])
            self.drop = nn.Dropout(dropout)
            self.pred = nn.Linear(emb, 1)

        def forward(self, q, r, qn, ctx):
            L = q.shape[1]
            x = self.inter(q + num_c * r) + self.pos(torch.arange(L, device=q.device))[None]
            h = self.ex(qn)
            k = torch.arange(L, device=q.device)
            # query j may attend inputs 0..ctx[j] (causal, and never the target's own attempt); where ctx < 0 the
            # output is not used, so it attends 0..j only to stay finite
            lim = torch.where(ctx >= 0, ctx, k[None].expand_as(ctx))
            mask = (k[None, None, :] > lim[:, :, None]).repeat_interleave(heads, dim=0)
            for b in self.blocks:
                h = b(h, x, mask)
            return torch.sigmoid(self.pred(self.drop(h))).squeeze(-1)

    return DKT() if kind == "dkt" else SAKT()


def windows(seqs, seq_len: int, upto_split: int):
    """Non-overlapping training windows over the rows with split <= upto_split."""
    out = []
    for _, c, r, s, _, g in seqs:
        keep = s <= upto_split
        c, r, g = c[keep], r[keep], g[keep]
        for a in range(0, max(len(c) - 1, 0), seq_len - 1):
            cc, rr, gg = c[a:a + seq_len], r[a:a + seq_len], g[a:a + seq_len]
            if len(cc) >= 2:
                out.append((cc, rr, gg))
    return out


def pad(batch, seq_len: int):
    """Inputs 0..n-2, targets 1..n-1; ``m`` marks targets that have a context (not padding, not first attempt)."""
    L = seq_len - 1
    q = np.zeros((len(batch), L), np.int64); r = np.zeros_like(q); qn = np.zeros_like(q); y = np.zeros_like(q)
    ctx = np.full((len(batch), L), -1, np.int64)
    for i, (c, a, g) in enumerate(batch):
        n = len(c) - 1
        q[i, :n], r[i, :n], qn[i, :n], y[i, :n] = c[:-1], a[:-1], c[1:], a[1:]
        ctx[i, :n] = context_index(g)
    return q, r, qn, y, ctx >= 0, ctx


def predict(model, seqs, seq_len: int, splits: tuple[int, ...], device: str, fallback: float, batch: int = 256):
    """Prequential predictions for the rows whose split is in ``splits``: windows of ``seq_len`` with stride
    ``seq_len // 2``; a position takes its prediction from the first window where it has at least ``seq_len // 2``
    earlier answers as context (or from the learner's first window). A row with no earlier attempt of the learner
    gets ``fallback`` (the mean answer of the training rows)."""
    import torch
    half = seq_len // 2
    jobs = []  # (c, r, g of the window, output positions to keep, row ids, answers)
    rows_out, p_out, y_out = [], [], []
    for _, c, r, s, rows, g in seqs:
        want = np.isin(s, splits)
        if not want.any():
            continue
        n = len(c)
        done = np.zeros(n, bool)
        start = 0
        while start < n - 1:
            end = min(start + seq_len, n)
            lo = 1 if start == 0 else start + half
            idx = np.arange(max(lo, start + 1), end)
            ctx = context_index(g[start:end])
            idx = idx[want[idx] & ~done[idx] & (ctx[idx - start - 1] >= 0)]
            if len(idx):
                jobs.append((c[start:end], r[start:end], g[start:end], idx - start - 1, rows[idx], r[idx]))
                done[idx] = True
            if end == n:
                break
            start += half
        rest = np.flatnonzero(want & ~done)
        if len(rest):
            rows_out.append(rows[rest]); p_out.append(np.full(len(rest), fallback)); y_out.append(r[rest])
    model.eval()
    with torch.no_grad():
        for a in range(0, len(jobs), batch):
            jb = jobs[a:a + batch]
            q, r, qn, _, _, ctx = pad([(j[0], j[1], j[2]) for j in jb], seq_len)
            t = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
            p = model(t(q), t(r), t(qn), t(ctx)).cpu().numpy()
            for i, j in enumerate(jb):
                p_out.append(p[i, j[3]]); rows_out.append(j[4]); y_out.append(j[5])
    return np.concatenate(rows_out), np.concatenate(p_out), np.concatenate(y_out)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def run(df: pd.DataFrame, kind: str, seed: int, emb: int = 64, seq_len: int = 200, heads: int = 8,
        dropout: float = 0.2, lr: float = 1e-3, batch: int = 64, max_epochs: int = 50, patience: int = 3):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs, num_c = sequences(df)
    train = windows(seqs, seq_len, 0)
    fallback = float(df.loc[df["split"] == "history", "correct"].mean())
    model = build_model(kind, num_c, emb, seq_len, heads, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    best, state, bad, curve, t0 = -1.0, None, 0, {}, time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for a in range(0, len(order), batch):
            q, r, qn, y, m, ctx = pad([train[i] for i in order[a:a + batch]], seq_len)
            t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
            p = model(t(q), t(r), t(qn), t(ctx))
            mk = t(m)
            loss = bce(p[mk], t(y).float()[mk])
            opt.zero_grad(); loss.backward(); opt.step()
        _, pv, yv = predict(model, seqs, seq_len, (1,), dev, fallback)
        a_v = auc(yv, pv)
        curve[ep] = round(a_v, 5)
        print(f"{kind} epoch {ep}: validation AUC {a_v:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if a_v > best:
            best, bad = a_v, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    rows, p, y = predict(model, seqs, seq_len, (1, 2), dev, fallback)
    split = np.empty(len(df), dtype=object)
    split[:] = df["split"].to_numpy()
    pred = pd.DataFrame({"row": rows, "user_id": df["user_id"].to_numpy()[rows], "split": split[rows], "y": y,
                         kind: p}).sort_values("row").reset_index(drop=True)
    choice = {"model": kind, "epochs": int(max(curve, key=curve.get)), "val_auc_by_epoch": curve, "num_skills": num_c,
              "emb": emb, "seq_len": seq_len, "heads": heads if kind == "sakt" else None, "dropout": dropout, "lr": lr,
              "batch": batch, "train_windows": len(train), "device": dev}
    return pred, choice


def scores(pred: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for sp, g in pred.groupby("split"):
        y, p = g["y"].to_numpy(dtype=float), np.clip(g[kind].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        rows.append({"model": kind, "split": sp, "AUC": auc(y, p),
                     "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                     "RMSE": float(np.sqrt(((y - p) ** 2).mean())), "n": len(g)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--model", required=True, choices=["dkt", "sakt"])
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    df = pd.read_parquet(out_root / f"interactions_{args.dataset}.parquet")
    pred, choice = run(df, args.model, seed, batch=args.batch, max_epochs=args.max_epochs)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_root / "predictions" / f"b1_deep_{args.model}_{args.dataset}.parquet", index=False)
    sc = scores(pred, args.model)
    sc.round(4).to_csv(tables / f"b1_deep_{args.model}_{args.dataset}.csv", index=False)
    (tables / f"b1_deep_{args.model}_choices_{args.dataset}.json").write_text(json.dumps(choice, indent=2),
                                                                              encoding="utf-8")
    print(sc.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
