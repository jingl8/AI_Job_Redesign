#!/usr/bin/env python3
"""AI exposure and position in the occupational handoff network.

Two questions, at whatever scope the codes file covers (the SOC 11/13/43 pilot or all O*NET occupations):

  A. Which kinds of nodes are more or less exposed to AI? Nodes are typed by the community
     algorithms (Louvain and Leiden consensus clusters of the main-spec handoff network) and by their
     position: participation coefficient (boundary spanning), within-cluster strength z (centrality
     inside the cluster), Guimera-Amaral role, weighted betweenness (brokerage), total handoff
     strength, and producer-consumer balance.

  B. Do handoffs predict exposure better than occupational similarity? The handoff network is compared
     with four similarity networks built from O*NET: work activities, skills, task content (detailed
     work activities), and O*NET's curated related occupations. Tests:
       B1 network autocorrelation (Moran's I) with permutation p-values;
       B2 out-of-sample prediction of each occupation's exposure from its neighbours' exposure
          (k-nearest-neighbour lags, repeated 10-fold CV), alone and with the handoff lag added,
          with bootstrap CIs for the gains;
       B3 dyadic MRQAP: exposure similarity of each pair on handoff flow and similarity, QAP p-values;
       B4 variance explained by clusters: handoff clusters vs similarity-network clusters with the
          same number of clusters vs SOC minor groups;
       B5 direction: exposure of upstream producers vs downstream consumers.

  C. H1: How does occupational similarity interact with handoffs in relation to AI exposure?
       C1 dyadic MRQAP with a handoff x similarity interaction, for each similarity network and two
          codings of handoffs (log flow; any tie), QAP p-values;
       C2 exposure gaps |e_i - e_j| by handoff tie x similarity (below/above median), with a
          permutation test of the difference in tie effects between similar and dissimilar pairs;
       C3 node level: does the exposure of an occupation's handoff partners predict its own exposure
          more when those partners are similar to it (lag x partner-similarity moderation)?

  D. H2: Are occupations involving human decision making more or less exposed?
       D1 decision-making measures: O*NET "Making Decisions and Solving Problems" (work activity),
          "Freedom to Make Decisions", "Frequency of Decision Making", "Impact of Decisions on
          Co-workers or Company Results" (work context) and, when the full Skills file is supplied,
          "Judgment and Decision Making"; their standardised mean (decision index, with Cronbach's
          alpha); and, from the handoff codes, the time-weighted share of outputs coded "decide";
       D2 occupation-level regressions (bivariate; + job zone and employment; + cluster fixed effects;
          quadratic and quintile checks);
       D3 task level within occupations: do tasks with a "decide" output carry lower GPT-4 or human
          exposure labels than other tasks of the same occupation (occupation fixed effects,
          occupation-clustered standard errors)?

Exposure measures (all public):
  gpt4_beta        Eloundou et al. (2024), GPT-4 rated, occupation level (primary)
  human_beta       Eloundou et al. (2024), human rated, occupation level
  gpt4_beta_tw     Eloundou et al. task labels (E1 = 1, E2 = .5, E0 = 0) averaged with the handoff
                   network's own O*NET time weights
  aioe             Felten, Raj & Seamans (2021) AI Occupational Exposure (2010 SOC; partial match)
Files are downloaded from GitHub into --exposure-dir on first use, or read from there if present.

Caution built into the design: Eloundou et al. rated exposure from O*NET task statements, and the
task-content similarity network is built from the same statements, so it predicts exposure partly by
construction. Work-activity and skill similarity use separate O*NET ratings and are the cleaner
comparisons. The handoff network is also task-derived but records relations between occupations,
not their content.

Usage:
    python handoff_exposure.py --data-dir data --code-dir code --out-dir exposure_out \
        [--communities-out handoff_communities_out]   # reuse the clusters from that run
Requires the packages of handoff_communities.py; it imports that script from --code-dir or this folder.
"""
import argparse
import importlib.util
import os
import sys
import time
import urllib.request
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

URLS = {
    "occ_level.csv": "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/occ_level.csv",
    "full_labelset.tsv": "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/full_labelset.tsv",
    "AIOE_DataAppendix.xlsx": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx",
}
MEASURES = {"gpt4_beta": "GPT-4 beta (Eloundou et al.)", "human_beta": "Human beta (Eloundou et al.)",
            "gpt4_beta_tw": "GPT-4 beta, time-weighted", "aioe": "AIOE (Felten et al.)"}
SIM_NETS = ["work_activities", "skills", "task_content", "related_occupations"]
NET_LABEL = {"handoff": "Handoffs", "work_activities": "Work-activity similarity", "skills": "Skill similarity",
             "task_content": "Task-content similarity", "related_occupations": "O*NET related occupations"}
TIER = {"Primary-Short": 3.0, "Primary-Long": 2.0, "Supplemental": 1.0}


# ----------------------------------------------------------------------------- helpers
def load_communities_module(dirs):
    for d in dirs:
        p = os.path.join(d, "handoff_communities.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("handoff_communities", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, p
    raise FileNotFoundError("handoff_communities.py not found in " + ", ".join(dirs))


def fetch(exposure_dir, name, log, required=True):
    os.makedirs(exposure_dir, exist_ok=True)
    path = os.path.join(exposure_dir, name)
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(URLS[name], path)
            log(f"Downloaded {name} -> {path}")
        except Exception as ex:
            if required:
                raise RuntimeError(f"Could not download {URLS[name]} ({ex}). Download it manually into {exposure_dir}.")
            log(f"WARNING: {name} unavailable ({ex}); measures that need it are skipped.")
            return None
    return path


def zscore(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def rownorm(W):
    s = W.sum(1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def knn(W, k):
    """Keep each row's k largest positive weights (directed; row i = i's neighbours)."""
    W = np.array(W, float)
    np.fill_diagonal(W, 0)
    out = np.zeros_like(W)
    idx = np.argsort(-W, axis=1)[:, :k]
    rows = np.arange(len(W))[:, None]
    vals = W[rows, idx]
    out[rows, idx] = np.where(vals > 0, vals, 0)
    return out


def corr_similarity(X):
    """Pearson correlation between occupations' standardised profiles; negatives set to 0."""
    X = np.asarray(X, float)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    Z = Z - Z.mean(1, keepdims=True)
    Z = Z / np.where(np.linalg.norm(Z, axis=1, keepdims=True) > 0, np.linalg.norm(Z, axis=1, keepdims=True), 1)
    S = Z @ Z.T
    np.fill_diagonal(S, 0)
    return np.clip(S, 0, None)


def ols_hc3(y, X, names):
    X = np.column_stack([np.ones(len(y)), X])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    V = XtXi @ (X.T * (r / np.clip(1 - h, 1e-8, None)) ** 2) @ X @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    n, p = X.shape
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    from scipy.stats import norm
    tab = pd.DataFrame({"beta": b, "se_hc3": se, "t": b / np.where(se > 0, se, np.nan)}, index=["const"] + names)
    tab["p"] = 2 * norm.sf(np.abs(tab.t))
    return tab, r2, 1 - (1 - r2) * (n - 1) / (n - p)


def vif(X, names):
    out = {}
    for j, nm in enumerate(names):
        others = np.delete(X, j, axis=1)
        _, r2, _ = ols_hc3(X[:, j], others, [f"x{i}" for i in range(others.shape[1])])
        out[nm] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


def eta2(labels, y):
    df = pd.DataFrame({"y": y, "g": labels})
    ssb = df.groupby("g").y.apply(lambda s: len(s) * (s.mean() - y.mean()) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    k, n = df.g.nunique(), len(y)
    e2 = ssb / sst
    return e2, 1 - (1 - e2) * (n - 1) / (n - k), k


def moran(W, x):
    z = x - x.mean()
    Wn = rownorm(W)
    return (len(x) / Wn.sum()) * (z @ Wn @ z) / (z @ z) if Wn.sum() > 0 else np.nan


def moran_perm(W, x, n_perm, rng):
    obs = moran(W, x)
    Wn = rownorm(W)
    S0 = Wn.sum()
    null = np.empty(n_perm)
    for i in range(n_perm):
        z = rng.permutation(x - x.mean())
        null[i] = (len(x) / S0) * (z @ Wn @ z) / (z @ z)
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1), null.mean()


def cv_predict(X, y, reps, seed=0):
    """Repeated 10-fold CV of a linear model; returns mean R2 and per-observation mean squared error."""
    X = np.asarray(X, float).reshape(len(y), -1)
    se = np.zeros(len(y))
    r2s = []
    for r in range(reps):
        pred = np.zeros(len(y))
        for tr, te in KFold(10, shuffle=True, random_state=seed + r).split(X):
            Xt = np.column_stack([np.ones(len(tr)), X[tr]])
            b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
            pred[te] = np.column_stack([np.ones(len(te)), X[te]]) @ b
        err = (y - pred) ** 2
        se += err / reps
        r2s.append(1 - err.sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(r2s)), se


def boot_delta_r2(se_a, se_b, y, n_boot, rng):
    """R2 gain of model b over model a (paired over occupations) with a bootstrap 95% CI."""
    var = ((y - y.mean()) ** 2).mean()
    d = (se_a - se_b) / var
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def mrqap(y, dyad_X, names, n_perm, rng):
    """OLS of dyadic exposure similarity -|e_i - e_j| on dyadic predictors (all standardised);
    QAP p-values from permuting exposure across occupations."""
    n = len(y)
    iu = np.triu_indices(n, 1)
    X = np.column_stack([np.ones(len(iu[0]))] + [zscore(x) for x in dyad_X])
    proj = np.linalg.pinv(X.T @ X) @ X.T
    Y = zscore(-np.abs(y[:, None] - y[None, :])[iu])
    b = proj @ Y
    null = np.empty((n_perm, X.shape[1]))
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = proj @ zscore(-np.abs(yp[:, None] - yp[None, :])[iu])
    p = [(np.sum(np.abs(null[:, j]) >= abs(b[j])) + 1) / (n_perm + 1) for j in range(X.shape[1])]
    r2 = 1 - ((Y - X @ b) ** 2).sum() / (Y ** 2).sum()
    return pd.DataFrame({"std_beta": b[1:], "qap_p": p[1:], "null_2.5": np.percentile(null[:, 1:], 2.5, axis=0),
                         "null_97.5": np.percentile(null[:, 1:], 97.5, axis=0)}, index=names), r2


def leiden_matched_k(W, target_k, search_seeds=3, final_seeds=10, max_steps=15):
    """Leiden (undirected modularity) on a similarity network with the resolution bisected (log scale,
    gamma in [0.1, 10]) so the number of clusters of size > 1 matches target_k as closely as possible.
    The search uses a few seeds per step; the chosen resolution is then run with final_seeds."""
    import igraph as ig
    import leidenalg as la
    g = ig.Graph.Weighted_Adjacency(np.maximum(W, W.T).tolist(), mode="max", attr="weight", loops=False)

    def run(gamma, seeds):
        best = max((la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                      resolution_parameter=gamma, seed=s, n_iterations=2) for s in range(seeds)),
                   key=lambda p: p.quality())
        lab = np.array(best.membership)
        return lab, int((pd.Series(lab).value_counts() > 1).sum())

    lo, hi = 0.1, 10.0
    best = None
    for _ in range(max_steps):
        mid = np.sqrt(lo * hi)
        lab, k = run(mid, search_seeds)
        if best is None or abs(k - target_k) < abs(best[1] - target_k):
            best = (lab, k, mid)
        if k == target_k:
            break
        lo, hi = (mid, hi) if k < target_k else (lo, mid)
    lab, k = run(best[2], final_seeds)
    return lab, k, best[2]


# ------------------------------------------------------------------------------ data
def load_exposure(occs, pilot_tasks, exposure_dir, log):
    occ = pd.read_csv(fetch(exposure_dir, "occ_level.csv", log)).set_index("O*NET-SOC Code")
    ex = pd.DataFrame(index=pd.Index(occs, name="soc"))
    ex["gpt4_beta"] = occ.reindex(occs)["dv_rating_beta"].values
    ex["human_beta"] = occ.reindex(occs)["human_rating_beta"].values
    labels = pd.read_csv(fetch(exposure_dir, "full_labelset.tsv", log), sep="\t",
                         usecols=["Task ID", "beta", "human_exposure_agg"])
    labels = labels.dropna(subset=["Task ID"]).assign(**{"Task ID": lambda d: d["Task ID"].astype(int)})
    labels = labels.drop_duplicates("Task ID").set_index("Task ID")
    labels["human_beta_task"] = labels.human_exposure_agg.map({"E0": 0.0, "E1": 1.0, "E2": 0.5})
    lab = labels["beta"]
    t = pilot_tasks[["O*NET-SOC Code", "Task ID", "w"]].copy()
    t["beta"] = t["Task ID"].map(lab)
    t = t.dropna(subset=["beta"])
    ex["gpt4_beta_tw"] = (t.w * t.beta).groupby(t["O*NET-SOC Code"]).sum() / t.w.groupby(t["O*NET-SOC Code"]).sum()
    ex["task_label_coverage"] = (t.groupby("O*NET-SOC Code").size()
                                 / pilot_tasks.groupby("O*NET-SOC Code").size()).reindex(occs).values
    p = fetch(exposure_dir, "AIOE_DataAppendix.xlsx", log, required=False)
    if p:
        a = pd.read_excel(p, sheet_name="Appendix A")
        a["SOC Code"] = a["SOC Code"].astype(str).str.strip()
        ex["aioe"] = [a.set_index("SOC Code")["AIOE"].get(o[:7], np.nan) for o in occs]
    return ex, labels[["beta", "human_beta_task"]].rename(columns={"beta": "gpt4_beta_task"})


def similarity_networks(occs, files, pilot_tasks, log):
    n = len(occs)
    ix = {c: i for i, c in enumerate(occs)}
    nets, have = {}, {}
    for key, fname in [("work_activities", "wa"), ("skills", "sk")]:
        df = pd.read_excel(files[fname], usecols=["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"])
        prof = df[df["Scale ID"] == "IM"].pivot_table(index="O*NET-SOC Code", columns="Element ID",
                                                     values="Data Value").reindex(occs)
        ok = prof.notna().all(axis=1).values
        if key == "skills" and prof.shape[1] <= 10:
            NET_LABEL["skills"] = "Basic-skill similarity"
            log(f"  NOTE: the skills file has only {prof.shape[1]} elements (O*NET basic skills); supply the full "
                f"Skills.xlsx (35 skills) for full skill similarity and the Judgment and Decision Making item.")
        S = np.zeros((n, n))
        S[np.ix_(ok, ok)] = corr_similarity(prof.values[ok])
        nets[key], have[key] = S, ok
    dwa = pd.read_excel(files["dwa"], usecols=["Task ID", "DWA Element ID"]).drop_duplicates()
    d = dwa.merge(pilot_tasks[["O*NET-SOC Code", "Task ID", "w"]], on="Task ID")
    d["w"] = d.w / d.groupby("Task ID")["DWA Element ID"].transform("size")
    D = d.groupby(["O*NET-SOC Code", "DWA Element ID"]).w.sum().unstack(fill_value=0).reindex(occs).fillna(0).values
    nrm = np.linalg.norm(D, axis=1, keepdims=True)
    Dn = D / np.where(nrm > 0, nrm, 1)
    S = Dn @ Dn.T
    np.fill_diagonal(S, 0)
    nets["task_content"], have["task_content"] = S, nrm[:, 0] > 0
    rel = pd.read_excel(files["rel"], usecols=["O*NET-SOC Code", "Related O*NET-SOC Code", "Relatedness Tier"])
    R = np.zeros((n, n))
    for a, b, tier in rel.itertuples(index=False):
        if a in ix and b in ix:
            R[ix[a], ix[b]] = max(R[ix[a], ix[b]], TIER.get(tier, 1.0))
    R = np.maximum(R, R.T)
    nets["related_occupations"], have["related_occupations"] = R, R.sum(1) > 0
    for k in nets:
        log(f"  {NET_LABEL[k]}: {int(have[k].sum())} of {n} occupations have a profile; "
            f"{(nets[k][np.ix_(have[k], have[k])] > 0).mean():.1%} of pairs positive")
    return nets, have


def fe_ols(y, X, groups):
    """OLS with group fixed effects (within transformation) and CR1 group-clustered standard errors."""
    X = np.asarray(X, float).reshape(len(y), -1)
    df = pd.DataFrame(np.column_stack([y, X]))
    dm = (df - df.groupby(np.asarray(groups)).transform("mean")).values
    yy, XX = dm[:, 0], dm[:, 1:]
    XtXi = np.linalg.pinv(XX.T @ XX)
    b = XtXi @ XX.T @ yy
    r = yy - XX @ b
    G = np.asarray(groups)
    meat = np.zeros((XX.shape[1], XX.shape[1]))
    for g in np.unique(G):
        m = G == g
        sc = XX[m].T @ r[m]
        meat += np.outer(sc, sc)
    nG, (n, k) = len(np.unique(G)), XX.shape
    V = nG / (nG - 1) * (n - 1) / max(1, n - k - nG) * XtXi @ meat @ XtXi
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def tie_similarity_cells(y, tie, high_sim, n_perm, rng):
    """Mean exposure gaps by tie x similarity half; DiD = tie effect among similar pairs minus tie
    effect among dissimilar pairs (tie effect = gap without tie - gap with tie). Permutation of
    exposure across occupations gives the null."""
    n = len(y)
    iu = np.triu_indices(n, 1)

    def stats(v):
        gap = np.abs(v[:, None] - v[None, :])[iu]
        cells = {(t, h): gap[(tie == t) & (high_sim == h)].mean() for t in (False, True) for h in (False, True)}
        te_low = cells[(False, False)] - cells[(True, False)]
        te_high = cells[(False, True)] - cells[(True, True)]
        return cells, te_low, te_high, te_high - te_low

    cells, te_low, te_high, did = stats(y)
    null = np.array([stats(rng.permutation(y))[3] for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(did)) + 1) / (n_perm + 1)
    counts = {(t, h): int(((tie == t) & (high_sim == h)).sum()) for t in (False, True) for h in (False, True)}
    return cells, counts, te_low, te_high, did, p


DECISION_ITEMS = [  # (column, file key, element ID, scale)
    ("wa_making_decisions", "wa", "4.A.2.b.1", "IM"),
    ("wc_freedom_to_decide", "wc", "4.C.3.a.4", "CX"),
    ("wc_decision_frequency", "wc", "4.C.3.a.2.b", "CX"),
    ("wc_decision_impact", "wc", "4.C.3.a.2.a", "CX"),
    ("sk_judgment_decision", "sk", "2.B.4.e", "IM"),
]


def decision_measures(occs, onet, pilot_tasks, codes, log):
    """O*NET decision-making items, their standardised mean, and the code-based decide share."""
    frames = {}
    for key in ("wa", "wc", "sk"):
        frames[key] = pd.read_excel(onet[key], usecols=["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"])
    dm = pd.DataFrame(index=pd.Index(occs, name="soc"))
    for col, key, eid, scale in DECISION_ITEMS:
        f = frames[key]
        v = f[(f["Element ID"] == eid) & (f["Scale ID"] == scale)].groupby("O*NET-SOC Code")["Data Value"].mean()
        if v.empty:
            log(f"  {col}: not in the supplied file; left out of the decision index")
            continue
        dm[col] = v.reindex(occs).values
    items = [c for c in dm.columns]
    Z = dm[items].apply(lambda c: (c - c.mean()) / c.std())
    R = Z.corr().values
    rbar = R[np.triu_indices(len(items), 1)].mean()
    alpha = len(items) * rbar / (1 + (len(items) - 1) * rbar)
    dm["decision_index"] = Z.mean(axis=1, skipna=False)
    rows = []
    for r in pilot_tasks[["O*NET-SOC Code", "Task ID", "w"]].itertuples(index=False):
        c = codes.get(int(r[1]))
        outs = c["outputs"] if c else []
        acts = [o.get("action") for o in outs]
        rows.append({"occ": r[0], "task": int(r[1]), "w": r[2], "n_out": len(outs),
                     **{f"has_{a}": float(a in acts) for a in ("create", "update", "decide", "transmit")},
                     "decide_share": (sum(a == "decide" for a in acts) / len(outs)) if outs else 0.0})
    ta = pd.DataFrame(rows)
    dm["codes_decide_share"] = (ta.w * ta.decide_share).groupby(ta.occ).sum().div(ta.w.groupby(ta.occ).sum()).reindex(occs).values
    dm["codes_decide_task_share"] = (ta.w * ta.has_decide).groupby(ta.occ).sum().div(ta.w.groupby(ta.occ).sum()).reindex(occs).values
    return dm, items, alpha, ta


# ------------------------------------------------------------------------------ figures
def fig_exposure_by_cluster(ex, lab, names, measure, out):
    df = pd.DataFrame({"e": ex[measure].values, "c": lab})
    order = df.groupby("c").e.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 1.6))
    for r, c in enumerate(order):
        v = df.e[df.c == c].dropna().values
        ax.boxplot(v, positions=[r], vert=False, widths=0.6, showfliers=False,
                   medianprops=dict(color="black"), boxprops=dict(color="0.5"), whiskerprops=dict(color="0.5"),
                   capprops=dict(color="0.5"))
        ax.scatter(v, r + np.random.default_rng(c).uniform(-0.18, 0.18, len(v)), s=12, alpha=0.7, color="tab:blue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"C{c + 1} {names[c]} (n={int((df.c == c).sum())})" for c in order], fontsize=8)
    ax.axvline(df.e.mean(), color="tab:red", ls="--", lw=0.8)
    ax.set_xlabel(MEASURES[measure])
    ax.set_title("AI exposure by handoff cluster (Louvain consensus); dashed = overall mean", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_network_exposure(H, A, lab, ex, measure, titles, occs, out):
    import networkx as nx
    pos, theta, radius = H.grouped_layout(A, lab)
    backbone = H.disparity_backbone(A, 0.05)
    G = nx.from_numpy_array(np.where(backbone, A, 0), create_using=nx.DiGraph)
    s = A.sum(0) + A.sum(1)
    fig, ax = plt.subplots(figsize=(11, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, edge_color="0.3", arrows=False, width=0.4)
    e = ex[measure].values
    sc = ax.scatter([pos[i][0] for i in range(len(occs))], [pos[i][1] for i in range(len(occs))],
                    s=20 + 500 * np.sqrt(s / s.max()), c=e, cmap="viridis", edgecolors="white", linewidths=0.5)
    for c, th in theta.items():
        r = 1 + radius[c] + 0.08
        ax.text(r * np.cos(th), r * np.sin(th), f"C{c + 1}", fontsize=11, fontweight="bold", ha="center", va="center")
    fig.colorbar(sc, ax=ax, shrink=0.6, label=MEASURES[measure])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Handoff network (disparity backbone), nodes coloured by AI exposure", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_position(df, coef_tabs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, x, lab in [(axes[0], "participation", "participation coefficient (boundary spanning)"),
                       (axes[1], "within_z", "within-cluster strength z (centrality in cluster)")]:
        ax.scatter(df[x], df.e, s=14, alpha=0.7)
        b = np.polyfit(df[x], df.e, 1)
        xs = np.linspace(df[x].min(), df[x].max(), 20)
        ax.plot(xs, np.polyval(b, xs), color="tab:red", lw=1)
        ax.set_xlabel(lab)
        ax.set_ylabel("AI exposure")
    ax = axes[2]
    names = [n for n in coef_tabs["M1 position"].index if n != "const"]
    for k, (m, tab) in enumerate(coef_tabs.items()):
        t = tab.reindex(names)
        ypos = np.arange(len(names)) + (k - (len(coef_tabs) - 1) / 2) * 0.25
        ax.errorbar(t.beta, ypos, xerr=1.96 * t.se_hc3, fmt="o", ms=4, capsize=2, label=m)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("standardised coefficient (95% CI, HC3)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_prediction(pred, gains, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    nets = list(dict.fromkeys(pred.network))
    ks = sorted(pred.k.unique(), key=lambda v: (v == "full", v if v != "full" else 0))
    w = 0.8 / len(ks)
    for j, k in enumerate(ks):
        d = pred[pred.k == k].set_index("network").reindex(nets)
        ax.bar(np.arange(len(nets)) + (j - (len(ks) - 1) / 2) * w, d.cv_r2, width=w, label=f"k = {k}")
    ax.set_xticks(range(len(nets)))
    short = {"handoff": "Handoffs", "work_activities": "Work\nactivities", "skills": "Skills",
             "task_content": "Task\ncontent", "related_occupations": "O*NET\nrelated"}
    ax.set_xticklabels([short.get(n, n) for n in nets], fontsize=8)
    ax.set_ylabel("out-of-sample R2 (repeated 10-fold CV)")
    ax.set_title("Predicting exposure from network neighbours' exposure", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    g = gains.reset_index(drop=True)
    ax.errorbar(g.delta_r2, np.arange(len(g)), xerr=[g.delta_r2 - g.ci_low, g.ci_high - g.delta_r2], fmt="o", capsize=3)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.comparison, fontsize=8)
    ax.set_xlabel("change in out-of-sample R2 (95% bootstrap CI)")
    ax.set_title("Head-to-head and incremental value of handoffs", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_mrqap(tab, out):
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(tab) + 1.5))
    yv = np.arange(len(tab))
    ax.hlines(yv, tab["null_2.5"], tab["null_97.5"], color="0.75", lw=6, label="95% QAP null")
    ax.plot(tab.std_beta, yv, "o", color="tab:blue", label="estimate")
    ax.set_yticks(yv)
    ax.set_yticklabels([NET_LABEL.get(i, i) for i in tab.index], fontsize=8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("standardised coefficient: exposure similarity of a pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_h1(y, flow, sim, mq_tab, out):
    """Exposure gap by handoff-flow decile within similarity terciles, and the interaction estimates."""
    n = len(y)
    iu = np.triu_indices(n, 1)
    gap = np.abs(y[:, None] - y[None, :])[iu]
    f, s = flow[iu], sim[iu]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    terc = np.quantile(s, [1 / 3, 2 / 3])
    groups = [(s <= terc[0], "least similar third"), ((s > terc[0]) & (s <= terc[1]), "middle third"),
              (s > terc[1], "most similar third")]
    for (m, lab), col in zip(groups, ["tab:blue", "tab:orange", "tab:green"]):
        ax.axhline(gap[m & (f == 0)].mean(), ls=":", lw=0.8, color=col)
        pos = m & (f > 0)
        q = pd.qcut(f[pos], 10, labels=False, duplicates="drop")
        means = pd.Series(gap[pos]).groupby(q).mean()
        ax.plot(means.index + 1, means.values, "o-", label=lab, color=col)
    ax.set_xlabel("handoff flow decile among pairs with a tie (dotted: pairs with no tie)")
    ax.set_ylabel("mean exposure gap |e_i - e_j|")
    ax.legend(fontsize=8, title="work-activity similarity", title_fontsize=8)
    ax = axes[1]
    t = mq_tab.reset_index()
    ax.hlines(np.arange(len(t)), t["null_2.5"], t["null_97.5"], color="0.8", lw=6)
    ax.plot(t.std_beta, np.arange(len(t)), "o")
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["label"], fontsize=7)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("standardised coefficient on exposure similarity (grey: 95% QAP null)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_h2(dd, measure, task_tab, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    ax.scatter(dd.decision_index, dd[measure], s=14, alpha=0.7)
    b = np.polyfit(dd.decision_index, dd[measure], 1)
    xs = np.linspace(dd.decision_index.min(), dd.decision_index.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color="tab:red", lw=1)
    ax.set_xlabel("O*NET decision-making index (mean of standardised items)")
    ax.set_ylabel(MEASURES[measure])
    ax = axes[1]
    q = pd.qcut(dd.decision_index, 5, labels=False)
    g = dd.groupby(q)[measure].agg(["mean", "std", "count"])
    ax.errorbar(g.index + 1, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]), fmt="o-", capsize=3)
    ax.set_xlabel("decision-making index quintile")
    ax.set_ylabel(f"mean {MEASURES[measure]} (95% CI)")
    ax = axes[2]
    t = task_tab.reset_index()
    ax.errorbar(t.coef, np.arange(len(t)), xerr=1.96 * t.se, fmt="o", capsize=3)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["label"], fontsize=7)
    ax.set_xlabel("difference in task exposure within occupations (95% CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/mnt/project")
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default="exposure_out")
    ap.add_argument("--communities-out", default=None,
                    help="output folder of handoff_communities.py; its filtered/nodes.csv supplies the clusters")
    ap.add_argument("--exposure-dir", default="exposure_data")
    ap.add_argument("--measure", default="gpt4_beta", choices=list(MEASURES))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="neighbour counts for the lag models")
    ap.add_argument("--k-main", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cv-reps", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=100, help="seeds if the clusters are recomputed")
    ap.add_argument("--soc-majors", nargs="+", default=None)
    ap.add_argument("--impute", choices=["median", "mean", "equal"], default="median")
    ap.add_argument("--h1-similarity", default="work_activities", choices=SIM_NETS,
                    help="similarity network used for the H1 cell table, moderation model and figure")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261019)

    H, hpath = load_communities_module([args.code_dir, os.path.dirname(os.path.abspath(__file__))])
    log = H.Reporter(os.path.join(args.out_dir, "summary.txt"))
    N, _ = H.import_module("network", [args.code_dir, args.data_dir])
    CP, _ = H.import_module("copresence", [args.code_dir, args.data_dir])
    cache = os.path.abspath(args.cache_dir or getattr(N, "BASE", "") or "handoff_cache")
    N.BASE = os.path.join(cache, "")
    dirs = [args.data_dir, args.code_dir]
    files = {"codes": H.find_file(dirs, H.variants("final_codes.csv")),
             "vocab": H.find_file(dirs, H.variants("artifact_vocabulary.csv")),
             "tasks": H.find_file(dirs, H.variants("Task Statements.xlsx")),
             "ratings": H.find_file(dirs, H.variants("Task Ratings.xlsx")),
             "oews": H.find_file(dirs, ["nat4d_M2025_dl.xlsx"])}
    H.build_cache(files, cache, args.impute, False, log, args.soc_majors)
    onet = {"wa": H.find_file(dirs, H.variants("Work Activities.xlsx")),
            "sk": H.find_file(dirs, H.variants("Skills.xlsx") + H.variants("Essential Skills.xlsx")),
            "dwa": H.find_file(dirs, H.variants("Tasks to DWAs.xlsx")),
            "rel": H.find_file(dirs, H.variants("Related Occupations.xlsx")),
            "jz": H.find_file(dirs, H.variants("Job Zones.xlsx")),
            "wc": H.find_file(dirs, H.variants("Work Context.xlsx"))}

    specs, occs, titles, P, Cm, art_names, art_domain, info = H.build_networks(N, CP, cache, lambda *a: None)
    A = specs["filtered"]["A"]
    n = len(occs)
    pilot_tasks, codes, voc = N.load()
    pilot_tasks = pilot_tasks.reset_index()
    log.section("AI EXPOSURE AND POSITION IN THE HANDOFF NETWORK")
    log(f"Scope: {len(pilot_tasks)} tasks, {n} occupations; main-spec (co-presence filtered) handoff network.")

    # ---- clusters
    if args.communities_out and os.path.exists(os.path.join(args.communities_out, "filtered", "nodes.csv")):
        nd = pd.read_csv(os.path.join(args.communities_out, "filtered", "nodes.csv")).set_index("soc").reindex(occs)
        if nd.louvain.isna().any():
            raise ValueError("--communities-out does not cover the same occupations as the current scope")
        lou, lei = nd.louvain.values.astype(int) - 1, nd.leiden.values.astype(int) - 1
        log(f"Clusters read from {args.communities_out}/filtered/nodes.csv")
    else:
        t0 = time.time()
        res = {a: H.ensemble(a, A, args.n_seeds, 0.5, True, log) for a in ["louvain", "leiden"]}
        lou = res["louvain"]["consensus"]
        lei = H.align(lou, res["leiden"]["consensus"])
        log(f"Clusters recomputed ({args.n_seeds} seeds per algorithm, {time.time() - t0:.0f}s)")
    names = {"louvain": H.cluster_names(lou, occs, titles), "leiden": H.cluster_names(lei, occs, titles)}
    Pc, z, role = H.participation(A, lou)
    s_out, s_in = A.sum(1), A.sum(0)
    tot = s_out + s_in
    import igraph as ig
    gd = ig.Graph.Weighted_Adjacency(A.tolist(), mode="directed", attr="weight", loops=False)
    gd.es["dist"] = [1.0 / w for w in gd.es["weight"]]
    btw = np.array(gd.betweenness(weights="dist", directed=True)) / max(1, (n - 1) * (n - 2))

    # ---- exposure and controls
    ex, task_labels = load_exposure(occs, pilot_tasks, args.exposure_dir, log)
    jz = pd.read_excel(onet["jz"]).set_index("O*NET-SOC Code")["Job Zone"].reindex(occs).values
    task_id_col = "Task ID"
    num = pd.Series(0.0, index=occs)
    den = pd.Series(0.0, index=occs)
    for r in pilot_tasks[["O*NET-SOC Code", task_id_col, "w"]].itertuples(index=False):
        c = codes.get(int(r[1]))
        items = (c["inputs"] + c["outputs"]) if c else []
        if items:
            num[r[0]] += r[2] * sum(x["party"] == "external" for x in items) / len(items)
            den[r[0]] += r[2]
    ext = (num / den.replace(0, np.nan)).values
    emp = info["employment"].emp.reindex(occs).values
    log("Exposure measures: " + "; ".join(f"{MEASURES[m]}: {int(ex[m].notna().sum())} of {n}"
                                         for m in MEASURES if m in ex))
    log(f"Correlations among measures:\n{ex[[m for m in MEASURES if m in ex]].corr().round(3).to_string()}")

    log("\nSimilarity networks:")
    sims, have = similarity_networks(occs, onet, pilot_tasks, log)
    nets = {"handoff": A + A.T, **sims}
    has_handoff = tot > 0

    node = pd.DataFrame({
        "title": titles.values, "louvain": lou + 1, "louvain_name": [names["louvain"][c] for c in lou],
        "leiden": lei + 1, "leiden_name": [names["leiden"][c] for c in lei],
        "participation": Pc, "within_z": z, "role": role, "betweenness": btw,
        "betweenness_rank": pd.Series(btw).rank(pct=True).values,
        "log_strength": np.log(np.where(tot > 0, tot, np.nan)),
        "producer_balance": np.where(tot > 0, (s_out - s_in) / np.where(tot > 0, tot, 1), np.nan),
        "external_share": ext, "job_zone": jz, "log_employment": np.log(np.where(emp > 0, emp, np.nan)),
        "employment": emp}, index=pd.Index(occs, name="soc")).join(ex)

    y_all = node[args.measure].values
    ok = ~np.isnan(y_all) & has_handoff
    for k in SIM_NETS:
        ok &= have[k]
    log(f"\nCommon analysis sample for {MEASURES[args.measure]}: {int(ok.sum())} of {n} occupations "
        f"(exposure observed, at least one handoff, and a profile in every similarity network).")
    y = y_all[ok]
    idx = np.where(ok)[0]

    # ================================================================= A: which nodes are exposed
    log.section("A. WHICH KINDS OF NODES ARE MORE OR LESS EXPOSED")
    cl_rows = []
    for algo, lab in [("louvain", lou), ("leiden", lei)]:
        e2, e2adj, K = eta2(lab[ok], y)
        null = [eta2(rng.permutation(lab[ok]), y)[0] for _ in range(args.n_perm)]
        log(f"{algo.title()} clusters explain {e2:.1%} of exposure variance (adjusted {e2adj:.1%}, K = {K}; "
            f"permutation p = {(np.sum(np.array(null) >= e2) + 1) / (args.n_perm + 1):.4f})")
        for c in range(lab.max() + 1):
            m = ok & (lab == c)
            if m.sum() == 0:
                continue
            v = y_all[m]
            wts = np.nan_to_num(emp[m])
            cl_rows.append({"algorithm": algo, "cluster": c + 1, "name": names[algo][c], "n": int(m.sum()),
                            "mean": v.mean(), "sd": v.std(ddof=1) if m.sum() > 1 else np.nan,
                            "ci_low": v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else np.nan,
                            "ci_high": v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else np.nan,
                            "employment_weighted_mean": np.average(v, weights=wts) if wts.sum() > 0 else np.nan,
                            "min": v.min(), "max": v.max(),
                            "most_exposed": H.short_title(node.title.values[m][np.argmax(v)], 40),
                            "least_exposed": H.short_title(node.title.values[m][np.argmin(v)], 40)})
    cl = pd.DataFrame(cl_rows)
    cl.to_csv(os.path.join(args.out_dir, "exposure_by_cluster.csv"), index=False)
    log.table(cl[cl.algorithm == "louvain"].drop(columns="algorithm").sort_values("mean", ascending=False)
              .set_index("cluster"), "Exposure by Louvain cluster (sorted by mean)")

    rl = node[ok].groupby("role")[args.measure].agg(["count", "mean", "std"])
    rl.to_csv(os.path.join(args.out_dir, "exposure_by_role.csv"))
    log.table(rl, "Exposure by Guimera-Amaral role (Louvain partition)")

    pos_vars = ["participation", "within_z", "log_strength", "producer_balance", "betweenness_rank"]
    ctl_vars = ["external_share", "job_zone", "log_employment"]
    d = node[ok].dropna(subset=pos_vars + ctl_vars)
    yz = zscore(d[args.measure].values)
    Xp = np.column_stack([zscore(d[c].values) for c in pos_vars])
    Xc = np.column_stack([zscore(d[c].values) for c in pos_vars + ctl_vars])
    fe = pd.get_dummies(d.louvain.astype(str), drop_first=True).values.astype(float)
    tabs = {}
    rows = []
    for mname, X, nm in [("M1 position", Xp, pos_vars), ("M2 + content controls", Xc, pos_vars + ctl_vars),
                         ("M3 + cluster fixed effects", np.column_stack([Xc, fe]),
                          pos_vars + ctl_vars + [f"fe{i}" for i in range(fe.shape[1])])]:
        tab, r2, r2a = ols_hc3(yz, X, nm)
        tabs[mname] = tab.loc[[i for i in tab.index if not i.startswith("fe")]]
        log(f"\n{mname}: n = {len(yz)}, R2 = {r2:.3f}, adjusted R2 = {r2a:.3f}")
        log(tabs[mname].round(3).to_string())
        t = tabs[mname].copy()
        t["model"], t["r2"], t["adj_r2"], t["n"] = mname, r2, r2a, len(yz)
        rows.append(t)
    pd.concat(rows).to_csv(os.path.join(args.out_dir, "position_regression.csv"))
    v = pd.concat({"M1": vif(Xp, pos_vars), "M2": vif(Xc, pos_vars + ctl_vars)}, axis=1)
    log.table(v, "Variance inflation factors")
    from scipy.stats import spearmanr
    sp = pd.DataFrame([{"variable": c, "spearman_rho": spearmanr(d[c], d[args.measure]).statistic,
                        "p": spearmanr(d[c], d[args.measure]).pvalue} for c in pos_vars + ctl_vars]).set_index("variable")
    log.table(sp, "Bivariate Spearman correlations with exposure")
    sp.to_csv(os.path.join(args.out_dir, "position_spearman.csv"))

    # ================================================================= B: handoffs vs similarity
    log.section("B. DO HANDOFFS PREDICT EXPOSURE BETTER THAN OCCUPATIONAL SIMILARITY?")
    sub = {g: W[np.ix_(idx, idx)] for g, W in nets.items()}
    log("Overlap: correlation of handoff ties (log flow) with each similarity, across pairs:")
    iu = np.triu_indices(len(idx), 1)
    hd = np.log1p(sub["handoff"] / np.median(sub["handoff"][sub["handoff"] > 0]))
    for g in SIM_NETS:
        log(f"  {NET_LABEL[g]}: r = {np.corrcoef(hd[iu], sub[g][iu])[0, 1]:.3f}")

    mor = []
    for g, W in sub.items():
        for k in args.k + ["full"]:
            Wk = W.copy() if k == "full" else knn(W, k)
            I, p, e0 = moran_perm(Wk, y, args.n_perm, rng)
            mor.append({"network": g, "k": k, "moran_I": I, "perm_p": p, "null_mean": e0})
    mor = pd.DataFrame(mor)
    mor.to_csv(os.path.join(args.out_dir, "moran.csv"), index=False)
    log.table(mor.pivot(index="network", columns="k", values="moran_I").reindex(list(nets)),
              "B1 Moran's I of exposure (k nearest neighbours; 'full' = all weights)")

    lags, errs, pred_rows = {}, {}, []
    for k in args.k + ["full"]:
        for g, W in sub.items():
            Wk = W.copy() if k == "full" else knn(W, k)
            lags[(g, k)] = rownorm(Wk) @ y
            r2, se = cv_predict(lags[(g, k)], y, args.cv_reps)
            errs[(g, k)] = se
            pred_rows.append({"network": g, "k": k, "cv_r2": r2})
    pred = pd.DataFrame(pred_rows)
    km = args.k_main
    gains = []
    for g in SIM_NETS:
        r2b, seb = cv_predict(np.column_stack([lags[(g, km)], lags[("handoff", km)]]), y, args.cv_reps)
        pred_rows.append({"network": f"{g} + handoff", "k": km, "cv_r2": r2b})
        dlt, lo, hi = boot_delta_r2(errs[(g, km)], seb, y, args.n_boot, rng)
        gains.append({"comparison": f"{NET_LABEL[g]} + handoffs vs {NET_LABEL[g]} alone",
                      "delta_r2": dlt, "ci_low": lo, "ci_high": hi})
        dlt, lo, hi = boot_delta_r2(errs[(g, km)], errs[("handoff", km)], y, args.n_boot, rng)
        gains.append({"comparison": f"Handoffs vs {NET_LABEL[g]}", "delta_r2": dlt, "ci_low": lo, "ci_high": hi})
    allsim = np.column_stack([lags[(g, km)] for g in SIM_NETS])
    r2s, ses = cv_predict(allsim, y, args.cv_reps)
    r2sh, sesh = cv_predict(np.column_stack([allsim, lags[("handoff", km)]]), y, args.cv_reps)
    dlt, lo, hi = boot_delta_r2(ses, sesh, y, args.n_boot, rng)
    gains.append({"comparison": "All similarity + handoffs vs all similarity", "delta_r2": dlt, "ci_low": lo, "ci_high": hi})
    pred_rows += [{"network": "all similarity", "k": km, "cv_r2": r2s},
                  {"network": "all similarity + handoff", "k": km, "cv_r2": r2sh}]
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.out_dir, "network_prediction.csv"), index=False)
    gains = pd.DataFrame(gains)
    gains.to_csv(os.path.join(args.out_dir, "handoff_gains.csv"), index=False)
    log.table(pred.pivot(index="network", columns="k", values="cv_r2").reindex(list(nets)),
              f"B2 out-of-sample R2 of the neighbour-lag model ({args.cv_reps} x 10-fold CV)")
    log(f"All four similarity lags: R2 = {r2s:.3f}; with the handoff lag added: {r2sh:.3f}")
    log.table(gains.set_index("comparison"),
              f"Gains in out-of-sample R2 at k = {km} (positive = first model better; 95% bootstrap CI)")

    dy = [np.log1p(sub["handoff"] / np.median(sub["handoff"][sub["handoff"] > 0]))[iu]] + [sub[g][iu] for g in SIM_NETS]
    mq_rows = []
    for label, cols in [("handoff only", [0])] + [(f"{g} only", [j + 1]) for j, g in enumerate(SIM_NETS)] + \
                       [("joint", list(range(len(dy))))]:
        tab, r2 = mrqap(y, [dy[c] for c in cols], [(["handoff"] + SIM_NETS)[c] for c in cols], args.n_perm, rng)
        tab["model"], tab["r2"] = label, r2
        mq_rows.append(tab)
    mq = pd.concat(mq_rows)
    mq.to_csv(os.path.join(args.out_dir, "mrqap.csv"))
    log.table(mq, f"B3 MRQAP: pairwise exposure similarity on network ties (standardised; {args.n_perm} QAP permutations)")

    cv_rows = []
    target_k = int((pd.Series(lei[ok]).value_counts() > 1).sum())
    for algo, lab in [("handoff Louvain", lou[ok]), ("handoff Leiden", lei[ok])]:
        e2, e2a, K = eta2(lab, y)
        cv_rows.append({"partition": algo, "K": K, "eta2": e2, "adj_eta2": e2a})
    for g in SIM_NETS:
        lab, K, gam = leiden_matched_k(sub[g], target_k)
        e2, e2a, Kall = eta2(lab, y)
        cv_rows.append({"partition": f"{NET_LABEL[g]} (Leiden, gamma {gam:.2f})", "K": Kall, "eta2": e2, "adj_eta2": e2a})
    for lvl, nchar in [("SOC minor group", 4), ("SOC broad group", 6)]:
        e2, e2a, K = eta2(np.array([o[:nchar] for o in np.array(occs)[ok]]), y)
        cv_rows.append({"partition": lvl, "K": K, "eta2": e2, "adj_eta2": e2a})
    cvt = pd.DataFrame(cv_rows).set_index("partition")
    cvt.to_csv(os.path.join(args.out_dir, "cluster_variance.csv"))
    log.table(cvt, f"B4 exposure variance explained by partitions (similarity partitions matched to K = {target_k})")

    Ao = A[np.ix_(idx, idx)]
    dirn = []
    for lab, W in [("upstream (producers feeding the occupation)", Ao.T), ("downstream (consumers of its output)", Ao),
                   ("both directions", Ao + Ao.T)]:
        r2, _ = cv_predict(rownorm(W) @ y, y, args.cv_reps)
        dirn.append({"direction": lab, "moran_I": moran(W, y), "cv_r2": r2})
    dirn = pd.DataFrame(dirn).set_index("direction")
    dirn.to_csv(os.path.join(args.out_dir, "direction.csv"))
    log.table(dirn, "B5 Direction of handoffs (full flow weights)")

    # ---- robustness across exposure measures
    rob = []
    for m in [c for c in MEASURES if c in node]:
        yy_all = node[m].values
        okm = ok & ~np.isnan(yy_all)
        if okm.sum() < 20:
            continue
        yy = yy_all[okm]
        jj = np.where(okm)[0]
        row = {"measure": MEASURES[m], "n": int(okm.sum())}
        for g in nets:
            W = knn(nets[g][np.ix_(jj, jj)], km)
            row[f"R2 {g}"] = cv_predict(rownorm(W) @ yy, yy, args.cv_reps)[0]
        lh = rownorm(knn(nets["handoff"][np.ix_(jj, jj)], km)) @ yy
        lw = rownorm(knn(nets["work_activities"][np.ix_(jj, jj)], km)) @ yy
        row["dR2 handoff added to work activities"] = (cv_predict(np.column_stack([lw, lh]), yy, args.cv_reps)[0]
                                                     - row["R2 work_activities"])
        rob.append(row)
    rob = pd.DataFrame(rob).set_index("measure")
    rob.to_csv(os.path.join(args.out_dir, "robustness_measures.csv"))
    log.table(rob, f"Robustness: out-of-sample R2 of neighbour lags (k = {km}) by exposure measure")

    # ================================================================= C: H1 similarity x handoffs
    log.section("C. H1: HOW DOES OCCUPATIONAL SIMILARITY INTERACT WITH HANDOFFS?")
    flow_sym = sub["handoff"]
    logflow = np.log1p(flow_sym / np.median(flow_sym[flow_sym > 0]))
    anytie = (flow_sym > 0).astype(float)
    mq_rows, fig_rows = [], []
    for g in SIM_NETS:
        for coding, hmat in [("log flow", logflow), ("any tie", anytie)]:
            zh, zs_ = zscore(hmat[iu]), zscore(sub[g][iu])
            tab, r2 = mrqap(y, [zh, zs_, zh * zs_], ["handoff", "similarity", "handoff x similarity"], args.n_perm, rng)
            tab["similarity_network"], tab["handoff_coding"], tab["r2"] = g, coding, r2
            mq_rows.append(tab)
            if g == args.h1_similarity:
                for term, r in tab.iterrows():
                    fig_rows.append({"label": f"{term} ({coding})", **r[["std_beta", "null_2.5", "null_97.5"]].to_dict()})
    h1 = pd.concat(mq_rows)
    h1.to_csv(os.path.join(args.out_dir, "h1_interaction_mrqap.csv"))
    show = h1.reset_index().rename(columns={"index": "term"})
    show = show.pivot_table(index=["similarity_network", "handoff_coding"], columns="term",
                            values=["std_beta", "qap_p"]).round(4)
    log.table(show, f"C1 MRQAP: exposure similarity on handoff, similarity and their product "
                    f"(standardised; {args.n_perm} QAP permutations)")

    sim_main = sub[args.h1_similarity]
    hi = sim_main[iu] > np.median(sim_main[iu])
    flows = flow_sym[iu]
    cell_rows = []
    for tie_label, tie in [("any tie", flows > 0),
                           ("strong tie (top decile of positive flows)", flows > np.quantile(flows[flows > 0], 0.9))]:
        cells, counts, te_low, te_high, did, p = tie_similarity_cells(y, tie, hi, args.n_perm, rng)
        cell_rows.append({"tie definition": tie_label,
                          "gap: no tie, dissimilar": cells[(False, False)], "gap: tie, dissimilar": cells[(True, False)],
                          "gap: no tie, similar": cells[(False, True)], "gap: tie, similar": cells[(True, True)],
                          "tie effect, dissimilar": te_low, "tie effect, similar": te_high,
                          "difference (similar - dissimilar)": did, "perm_p": p,
                          "pairs (tie, similar)": counts[(True, True)], "pairs (tie, dissimilar)": counts[(True, False)]})
    cells = pd.DataFrame(cell_rows).set_index("tie definition")
    cells.to_csv(os.path.join(args.out_dir, "h1_cells.csv"))
    log.table(cells.T, f"C2 mean exposure gap |e_i - e_j| by handoff tie x {NET_LABEL[args.h1_similarity]} "
                       f"(split at the median); tie effect = reduction in the gap when a tie exists")

    Wh = rownorm(flow_sym)
    lag_h = Wh @ y
    partner_sim = (Wh * sim_main).sum(1)
    lag_s = rownorm(knn(sim_main, km)) @ y
    mod_rows = []
    for mname, X, nm in [
            ("handoff lag x partner similarity", np.column_stack([zscore(lag_h), zscore(partner_sim),
                                                                 zscore(lag_h) * zscore(partner_sim)]),
             ["handoff lag", "partner similarity", "handoff lag x partner similarity"]),
            ("+ similarity lag", np.column_stack([zscore(lag_h), zscore(partner_sim),
                                                 zscore(lag_h) * zscore(partner_sim), zscore(lag_s)]),
             ["handoff lag", "partner similarity", "handoff lag x partner similarity", "similarity lag"])]:
        tab, r2, r2a = ols_hc3(zscore(y), X, nm)
        tab["model"], tab["r2"], tab["adj_r2"] = mname, r2, r2a
        mod_rows.append(tab)
        log(f"\nC3 node-level moderation, {mname}: n = {len(y)}, R2 = {r2:.3f}")
        log(tab.drop(columns=["model", "r2", "adj_r2"]).round(3).to_string())
    pd.concat(mod_rows).to_csv(os.path.join(args.out_dir, "h1_node_moderation.csv"))
    node["handoff_partner_similarity"] = np.nan
    node.loc[node.index[idx], "handoff_partner_similarity"] = partner_sim
    fig_h1(y, flow_sym, sim_main, pd.DataFrame(fig_rows).set_index("label"),
           os.path.join(args.out_dir, "fig6_h1_interaction.png"))

    # ================================================================= D: H2 decision making
    log.section("D. H2: ARE OCCUPATIONS INVOLVING HUMAN DECISION MAKING MORE OR LESS EXPOSED?")
    dm, items, alpha, ta = decision_measures(occs, onet, pilot_tasks, codes, log)
    node = node.join(dm)
    log(f"Decision index = mean of standardised {', '.join(items)}; standardised Cronbach's alpha = {alpha:.3f}")
    log.table(dm.corr().round(3), "Correlations among decision-making measures")
    dvars = ["decision_index"] + items + ["codes_decide_share", "codes_decide_task_share"]
    dd = node.dropna(subset=[args.measure, "job_zone", "log_employment"] + dvars)
    log(f"Occupation-level sample: {len(dd)} occupations with exposure, all decision measures, job zone and employment.")
    yz = zscore(dd[args.measure].values)
    fe = pd.get_dummies(dd.louvain.astype(str), drop_first=True).values.astype(float)
    ctl = np.column_stack([zscore(dd.job_zone.values), zscore(dd.log_employment.values)])
    h2_rows = []
    for v_ in dvars:
        xv = zscore(dd[v_].values)[:, None]
        for mname, X in [("bivariate", xv), ("+ job zone, employment", np.column_stack([xv, ctl])),
                         ("+ cluster fixed effects", np.column_stack([xv, ctl, fe]))]:
            tab, r2, _ = ols_hc3(yz, X, [v_] + [f"c{i}" for i in range(X.shape[1] - 1)])
            h2_rows.append({"measure": v_, "model": mname, "beta": tab.beta[v_], "se_hc3": tab.se_hc3[v_],
                            "p": tab.p[v_], "r2": r2, "n": len(yz)})
    h2 = pd.DataFrame(h2_rows)
    h2.to_csv(os.path.join(args.out_dir, "h2_occupation_models.csv"), index=False)
    log.table(h2.pivot(index="measure", columns="model", values="beta").reindex(dvars)
              [["bivariate", "+ job zone, employment", "+ cluster fixed effects"]].round(3),
              f"D2 standardised effect of each decision measure on {MEASURES[args.measure]} (HC3)")
    log.table(h2.pivot(index="measure", columns="model", values="p").reindex(dvars)
              [["bivariate", "+ job zone, employment", "+ cluster fixed effects"]].round(4), "p-values")
    xq = zscore(dd.decision_index.values)
    tq, _, _ = ols_hc3(yz, np.column_stack([xq, xq ** 2, ctl]), ["decision_index", "decision_index^2", "jz", "emp"])
    log(f"Quadratic check (+ controls): linear {tq.beta['decision_index']:+.3f} (p = {tq.p['decision_index']:.3f}), "
        f"squared {tq.beta['decision_index^2']:+.3f} (p = {tq.p['decision_index^2']:.3f})")
    qt = dd.groupby(pd.qcut(dd.decision_index, 5, labels=[f"Q{i}" for i in range(1, 6)]), observed=True)[args.measure] \
        .agg(["count", "mean", "std"])
    log.table(qt.round(3), "Exposure by decision-index quintile")
    rob2 = []
    for m in [c for c in MEASURES if c in node]:
        d2 = node.dropna(subset=[m, "decision_index", "job_zone", "log_employment"])
        tab, _, _ = ols_hc3(zscore(d2[m].values), np.column_stack([zscore(d2.decision_index.values),
                                                                  zscore(d2.job_zone.values),
                                                                  zscore(d2.log_employment.values)]),
                            ["decision_index", "jz", "emp"])
        rob2.append({"measure": MEASURES[m], "n": len(d2), "beta": tab.beta["decision_index"],
                     "se_hc3": tab.se_hc3["decision_index"], "p": tab.p["decision_index"]})
    rob2 = pd.DataFrame(rob2).set_index("measure")
    rob2.to_csv(os.path.join(args.out_dir, "h2_robustness_measures.csv"))
    log.table(rob2, "Decision index (+ job zone, employment) by exposure measure")

    tt = ta.join(task_labels, on="task")
    task_rows = []
    for lab_col, lab_name in [("gpt4_beta_task", "GPT-4 task label"), ("human_beta_task", "human task label")]:
        t2 = tt.dropna(subset=[lab_col])
        b, se = fe_ols(t2[lab_col].values, t2[["has_decide", "has_update", "has_transmit"]].values, t2.occ.values)
        for j, a in enumerate(["decide", "update", "transmit"]):
            task_rows.append({"label": f"{lab_name}: task has a '{a}' output", "exposure_label": lab_name,
                              "action": a, "coef": b[j], "se": se[j], "n_tasks": len(t2),
                              "n_occupations": t2.occ.nunique(), "share_of_tasks": t2[f"has_{a}"].mean()})
    task_tab = pd.DataFrame(task_rows).set_index("label")
    task_tab.to_csv(os.path.join(args.out_dir, "h2_task_level.csv"))
    log.table(task_tab.drop(columns=["exposure_label", "action"]),
              "D3 task level: exposure of tasks by output action, within occupations "
              "(occupation fixed effects, occupation-clustered SEs; tasks may carry several actions)")
    fig_h2(dd, args.measure, task_tab, os.path.join(args.out_dir, "fig7_h2_decision_making.png"))

    node["lag_handoff"] = np.nan
    node.loc[node.index[idx], "lag_handoff"] = lags[("handoff", km)]
    for g in SIM_NETS:
        node[f"lag_{g}"] = np.nan
        node.loc[node.index[idx], f"lag_{g}"] = lags[(g, km)]
    node.to_csv(os.path.join(args.out_dir, "exposure_by_occupation.csv"))

    # ---- figures
    fig_exposure_by_cluster(node.loc[ok], lou[ok], names["louvain"], args.measure,
                            os.path.join(args.out_dir, "fig1_exposure_by_cluster.png"))
    fig_network_exposure(H, A, lou, node, args.measure, titles, occs,
                         os.path.join(args.out_dir, "fig2_network_exposure.png"))
    fig_position(d.assign(e=d[args.measure]), tabs, os.path.join(args.out_dir, "fig3_position.png"))
    fig_prediction(pred, gains, os.path.join(args.out_dir, "fig4_prediction.png"))
    fig_mrqap(mq[mq.model == "joint"], os.path.join(args.out_dir, "fig5_mrqap.png"))
    log(f"\nOutputs written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
