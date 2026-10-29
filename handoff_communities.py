#!/usr/bin/env python3
"""Community detection on the producer-consumer handoff network (SOC 11/13/43 pilot).

Builds the occupation-to-occupation handoff network with the project's own code
(network.py: P/C matrices and flows; copresence.py: OEWS industry co-presence filter),
then:
  1. reports descriptive and summary statistics of the weighted directed network;
  2. clusters it with Louvain (networkx) and Leiden (leidenalg), each run over many seeds
     and combined by consensus clustering (Lancichinetti & Fortunato 2012);
  3. compares the two partitions, characterises clusters (closure, carrying artifacts,
     participation coefficients / boundary jobs) and checks results against the figures
     recorded in handoff_network_notes.md;
  4. writes figures (PNG) and tables (CSV) for both the co-presence-filtered main
     specification and the unfiltered specification.

Inputs (looked up in --data-dir, then --code-dir; underscore and space variants accepted):
  final_codes.csv, artifact_vocabulary.csv   (or claude_handoff_*.csv)
  Task Statements.xlsx, Task Ratings.xlsx      (O*NET)
  nat4d_M2025_dl.xlsx                          (BLS OEWS May 2025, 4-digit NAICS)
  network.py, copresence.py                    (or claude_handoff_network.py, ...)

network.py and copresence.py read cached inputs (pilot_tasks.pkl, final_codes.json,
artifact_vocabulary.json, oews4d.pkl) from a scratch directory. This script rebuilds
those caches from the files above, then imports and calls the two modules unchanged.

Task weight w (pilot_tasks.pkl) follows stage 1 (claude/code/model.py, scheme "FIR"):
    w = RT/100 x IM x mean frequency category, normalised to sum to 1 within occupation.
146 of the 3,176 pilot tasks have no O*NET ratings (all tasks of Legislators,
Financial Risk Specialists, and Financial and Investment Analysts, plus a few in nine
other occupations). They get the occupation's median raw weight, or equal weights where
the whole occupation is unrated. The original stage-2 rule for these tasks was not saved;
this is a reconstruction (see --impute).

Requires: numpy, pandas, openpyxl, networkx>=3.0, python-igraph, leidenalg, scipy,
          scikit-learn, matplotlib
    pip install networkx igraph leidenalg scikit-learn matplotlib openpyxl

Usage:
    python handoff_communities.py --data-dir /mnt/project --code-dir /mnt/project \
        --out-dir ./handoff_communities_out
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from itertools import combinations
from unittest import mock

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# Hire-to-pay jobs as defined in stage 1 (claude/code/model.py)
HIRE_TO_PAY = {"11-3121.00": "HRM", "13-1071.00": "HRS", "43-4161.00": "HRA",
               "13-1141.00": "CBS", "43-3051.00": "PAY", "13-2011.00": "ACC (boundary)"}
HTP_CORE = ["11-3121.00", "13-1071.00", "43-4161.00", "13-1141.00", "43-3051.00"]

# Figures recorded in claude/handoff/handoff_network_notes.md (Louvain, resolution 1.0)
REPORTED = {
    "unfiltered": {"Q": 0.375, "within_share": 0.52, "top100_within": 90, "n_clusters": 9, "n_isolates": 1},
    "filtered": {"Q": 0.417, "within_share": 0.56, "n_clusters": None, "n_isolates": None},
    "ARI_filtered_vs_unfiltered": 0.88,
    "moved_by_filter": 8,
    "pairs_discounted": 0.47,
    "sizes_unfiltered": "27-27-23-21-19-15-13-6-3-1",
}

PARTY = {"i": "internal", "e": "external", "b": "either"}
ACTION = {"c": "create", "u": "update", "d": "decide", "t": "transmit"}
TOKEN = re.compile(r"^(?:(?P<id>A\d{3})|NEW\[(?P<new>[^\]]+)\]):(?P<party>[ieb])(?::(?P<act>[cudt]))?$")


# ----------------------------------------------------------------------------- utilities
class Reporter:
    """Print to stdout and mirror into summary.txt."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        s = " ".join(str(a) for a in args)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()

    def table(self, df, title=None, floatfmt="{:.4g}".format):
        if title:
            self("\n" + title)
            self("-" * len(title))
        self(df.to_string(float_format=floatfmt))

    def section(self, title):
        self("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)


def find_file(dirs, names):
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"None of {names} found in {dirs}")


def variants(name):
    """'Task Statements.xlsx' -> also 'Task_Statements.xlsx'; 'final_codes.csv' -> 'claude_handoff_final_codes.csv'."""
    v = [name, name.replace(" ", "_"), name.replace("_", " ")]
    v += ["claude_handoff_" + x for x in v]
    return list(dict.fromkeys(v))


def import_module(name, dirs):
    path = find_file(dirs, [f"{name}.py", f"claude_handoff_{name}.py"])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def short_title(t, n=30):
    t = re.sub(r",? Except .*$", "", t).replace(" Occupations", "").replace(" and ", " & ")
    return t if len(t) <= n else t[: n - 1].rstrip(" ,&") + "…"


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ------------------------------------------------------------------------- cache build
def parse_codes(s, is_output):
    out = []
    for tok in [t for t in str(s).split(";") if t.strip()]:
        m = TOKEN.match(tok.strip())
        if m is None:
            raise ValueError(f"Unparseable code token: {tok!r}")
        e = {"artifact_id": "NEW" if m.group("new") else m.group("id"), "party": PARTY[m.group("party")]}
        if m.group("new"):
            e["new_name"] = m.group("new")
        if is_output:
            e["action"] = ACTION.get(m.group("act"))
        out.append(e)
    return out


def build_cache(files, cache, impute, rebuild, log):
    os.makedirs(cache, exist_ok=True)
    targets = ["pilot_tasks.pkl", "final_codes.json", "artifact_vocabulary.json", "oews4d.pkl"]
    if not rebuild and all(os.path.exists(os.path.join(cache, t)) for t in targets):
        log(f"Using cached inputs in {cache} (pass --rebuild to regenerate).")
        return
    t0 = time.time()
    codes = pd.read_csv(files["codes"], keep_default_na=False, dtype={"inputs": str, "outputs": str})
    recs = [{"task_id": int(r.task_id), "inputs": parse_codes(r.inputs, False),
             "outputs": parse_codes(r.outputs, True), "verifier": r.verifier} for r in codes.itertuples()]
    json.dump(recs, open(os.path.join(cache, "final_codes.json"), "w"))
    voc = pd.read_csv(files["vocab"])
    voc["generic"] = voc["generic"].astype(str).str.lower().eq("true")
    voc.to_json(os.path.join(cache, "artifact_vocabulary.json"), orient="records")

    ids = set(codes.task_id)
    ts = pd.read_excel(files["tasks"], usecols=["O*NET-SOC Code", "Title", "Task ID", "Task", "Task Type"])
    pt = ts[ts["Task ID"].isin(ids)].copy()
    tr = pd.read_excel(files["ratings"], usecols=["Task ID", "Scale ID", "Category", "Data Value"])
    tr = tr[tr["Task ID"].isin(ids)]
    rt = tr[tr["Scale ID"] == "RT"].groupby("Task ID")["Data Value"].mean()
    im = tr[tr["Scale ID"] == "IM"].groupby("Task ID")["Data Value"].mean()
    ft = tr[tr["Scale ID"] == "FT"]
    ft_mean = (ft["Category"] * ft["Data Value"]).groupby(ft["Task ID"]).sum() / ft.groupby("Task ID")["Data Value"].sum()
    pt["RT"], pt["IM"], pt["FT_mean"] = pt["Task ID"].map(rt), pt["Task ID"].map(im), pt["Task ID"].map(ft_mean)
    pt["raw_w"] = pt.RT / 100 * pt.IM * pt.FT_mean
    pt["imputed"] = pt.raw_w.isna()
    if impute == "median":
        pt["raw_w"] = pt.raw_w.fillna(pt.groupby("O*NET-SOC Code").raw_w.transform("median"))
    elif impute == "mean":
        pt["raw_w"] = pt.raw_w.fillna(pt.groupby("O*NET-SOC Code").raw_w.transform("mean"))
    pt["raw_w"] = pt.raw_w.fillna(1.0)   # occupations with no rated task -> equal weights
    pt["w"] = pt.raw_w / pt.groupby("O*NET-SOC Code").raw_w.transform("sum")
    pt.reset_index(drop=True).to_pickle(os.path.join(cache, "pilot_tasks.pkl"))

    oews = pd.read_excel(files["oews"], dtype=str,
                         usecols=["NAICS", "NAICS_TITLE", "OCC_CODE", "OCC_TITLE", "O_GROUP", "TOT_EMP"])
    oews.to_pickle(os.path.join(cache, "oews4d.pkl"))
    log(f"Built caches in {cache} ({time.time() - t0:.0f}s): {len(pt)} tasks, "
        f"{pt['O*NET-SOC Code'].nunique()} occupations, {int(pt.imputed.sum())} tasks with imputed weights.")


def industry_matrix(CP, occs, oews_path):
    """copresence.industry_matrix hard-codes /home/claude/handoff/oews4d.pkl; redirect it if needed."""
    real = pd.read_pickle

    def redirected(path, *a, **k):
        return real(oews_path if str(path).endswith("oews4d.pkl") else path, *a, **k)

    with mock.patch.object(pd, "read_pickle", redirected):
        return CP.industry_matrix(occs)


# ---------------------------------------------------------------------- network build
def build_networks(N, CP, cache, log):
    p, codes, voc = N.load()
    P, Cm = N.matrices(p, codes, voc)
    F = N.flows(P, Cm)
    occs = list(F.index)
    M, tot, used = industry_matrix(CP, occs, os.path.join(cache, "oews4d.pkl"))
    E, Esym, W, weitz, S = CP.copresence(M, tot)
    W = W.reindex(index=occs, columns=occs)
    if W.isna().any().any():
        log("WARNING: occupations without OEWS rows get co-presence weight 1:",
            sorted(W.index[W.isna().all(1)]))
        W = W.fillna(1.0)
    FW = N.flows(P, Cm, E=W.values)
    off = ~np.eye(len(occs), dtype=bool)
    titles = p.groupby("O*NET-SOC Code")["Title"].first().reindex(occs)
    art_names = {a: (voc.loc[a, "name"] if a in voc.index else a.replace("NEW:", "NEW: ")) for a in P.columns}
    art_domain = {a: (voc.loc[a, "domain"] if a in voc.index else "new") for a in P.columns}
    info = {"n_tasks": len(p), "n_occ": len(occs), "n_artifacts": P.shape[1],
            "pairs_discounted": float((Esym.reindex(index=occs, columns=occs).values[off] < 1).mean()),
            "broad_matches": sorted(o for o, c in used.items() if c != o[:7] and c[:2].isdigit()),
            "special_matches": {o: c for o, c in used.items() if not c[:2].isdigit()},
            "n_imputed": int(p["imputed"].sum()) if "imputed" in p else None}
    mult_unf = np.ones((len(occs), len(occs)))
    np.fill_diagonal(mult_unf, 0)
    mult_f = W.values.copy()
    np.fill_diagonal(mult_f, 0)
    specs = {"filtered": {"A": FW.values, "mult": mult_f, "label": "co-presence filtered (main spec)"},
             "unfiltered": {"A": F.values, "mult": mult_unf, "label": "unfiltered"}}
    return specs, occs, titles, P, Cm, art_names, art_domain, info


# ------------------------------------------------------------------------- statistics
def disparity_backbone(A, alpha=0.05):
    """Directed disparity filter (Serrano, Boguna & Vespignani 2009): keep an edge if it is
    significant for the source's out-strength or the target's in-strength."""
    B = A > 0
    s_out, s_in = A.sum(1), A.sum(0)
    k_out, k_in = B.sum(1), B.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_out = np.where(B, A / s_out[:, None], 0)
        p_in = np.where(B, A / s_in[None, :], 0)
        a_out = np.where(k_out[:, None] > 1, (1 - p_out) ** (k_out[:, None] - 1), 0.0)
        a_in = np.where(k_in[None, :] > 1, (1 - p_in) ** (k_in[None, :] - 1), 0.0)
    return B & ((a_out < alpha) | (a_in < alpha))


def path_stats(A_bool):
    """Unweighted path statistics on the largest strongly connected component."""
    G = nx.from_numpy_array(A_bool.astype(int), create_using=nx.DiGraph)
    scc = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(scc)
    return len(scc), nx.average_shortest_path_length(H), nx.diameter(H)


def describe(A, occs, titles, backbone, log, spec_label):
    n = len(A)
    off = ~np.eye(n, dtype=bool)
    w = A[off]
    pos = w[w > 0]
    B = A > 0
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    Gu = nx.from_numpy_array(A + A.T)
    s_out, s_in = A.sum(1), A.sum(0)
    isolates = [occs[i] for i in range(n) if s_out[i] + s_in[i] == 0]
    wcc = list(nx.weakly_connected_components(G))
    scc = list(nx.strongly_connected_components(G))
    recip_unw = (B & B.T)[off].sum() / max(1, B[off].sum())
    recip_w = np.minimum(A, A.T)[off].sum() / A[off].sum()
    top1 = np.sort(pos)[::-1][: max(1, int(round(0.01 * len(pos))))].sum() / pos.sum()
    top10 = np.sort(pos)[::-1][: max(1, int(round(0.10 * len(pos))))].sum() / pos.sum()
    scc_n, apl, diam = path_stats(B)
    bscc_n, bapl, bdiam = path_stats(backbone)
    clus_w = nx.average_clustering(Gu, weight="weight")
    clus_u = nx.average_clustering(nx.from_numpy_array((B | B.T).astype(int)))
    trans_unw = nx.transitivity(nx.from_numpy_array((B | B.T).astype(int)))
    deg_assort = nx.degree_assortativity_coefficient(nx.from_numpy_array((B | B.T).astype(int)))
    str_assort = nx.degree_assortativity_coefficient(Gu, weight="weight")
    rows = [
        ("Nodes (occupations)", n),
        ("Isolates", len(isolates)),
        ("Directed edges (w > 0)", int(B[off].sum())),
        ("Density", B[off].mean()),
        ("Total flow (sum of w)", A.sum()),
        ("Edge weight: mean", pos.mean()),
        ("Edge weight: median", np.median(pos)),
        ("Edge weight: max", pos.max()),
        ("Edge weight: CV", pos.std() / pos.mean()),
        ("Edge weight: Gini", gini(pos)),
        ("Flow share in top 1% of edges", top1),
        ("Flow share in top 10% of edges", top10),
        ("Reciprocity (unweighted)", recip_unw),
        ("Reciprocity (weighted)", recip_w),
        ("Weakly connected components", len(wcc)),
        ("Strongly connected components", len(scc)),
        ("Largest SCC size", scc_n),
        ("Avg shortest path (unweighted, LSCC)", apl),
        ("Diameter (unweighted, LSCC)", diam),
        ("Avg clustering (unweighted, symmetrised)", clus_u),
        ("Avg clustering (Onnela, w scaled by max w; symmetrised)", clus_w),
        ("Transitivity (unweighted, symmetrised)", trans_unw),
        ("Degree assortativity (unweighted, symmetrised)", deg_assort),
        ("Strength assortativity (symmetrised)", str_assort),
        ("Out-strength: mean / median", f"{s_out.mean():.4f} / {np.median(s_out):.4f}"),
        ("In-strength: mean / median", f"{s_in.mean():.4f} / {np.median(s_in):.4f}"),
        ("Spearman(in-strength, out-strength)", pd.Series(s_in).corr(pd.Series(s_out), method="spearman")),
        ("Backbone (disparity, a=.05): edges", int(backbone.sum())),
        ("Backbone: share of edges kept", backbone.sum() / B[off].sum()),
        ("Backbone: share of flow kept", A[backbone].sum() / A.sum()),
        ("Backbone: largest SCC / avg path / diameter", f"{bscc_n} / {bapl:.2f} / {bdiam}"),
    ]
    stats = pd.DataFrame(rows, columns=["statistic", "value"]).set_index("statistic")
    log.table(stats, f"Network descriptive statistics: {spec_label}")
    if isolates:
        log("Isolates:", ", ".join(f"{o} {titles[o]}" for o in isolates))

    for u, v, d in G.edges(data=True):
        d["dist"] = 1.0 / d["weight"]
    btw = nx.betweenness_centrality(G, weight="dist", normalized=True)
    pr = nx.pagerank(G, weight="weight")
    hubs, auths = nx.hits(G, max_iter=2000, normalized=True)
    nodes = pd.DataFrame({
        "soc": occs, "title": titles.values,
        "out_strength": s_out, "in_strength": s_in, "total_strength": s_out + s_in,
        "out_degree": B.sum(1), "in_degree": B.sum(0),
        "net_producer": s_out - s_in,
        "betweenness_w": [btw[i] for i in range(n)], "pagerank": [pr[i] for i in range(n)],
        "hub": [hubs[i] for i in range(n)], "authority": [auths[i] for i in range(n)],
        "clustering_w": [nx.clustering(Gu, i, weight="weight") for i in range(n)],
    })
    summ = nodes[["out_strength", "in_strength", "total_strength", "out_degree", "in_degree",
                  "betweenness_w", "pagerank", "clustering_w"]].describe(percentiles=[.1, .25, .5, .75, .9]).T
    log.table(summ, "Node-level summary statistics")
    cols = ["soc", "title", "out_strength", "in_strength", "betweenness_w", "pagerank"]
    log.table(nodes.nlargest(10, "out_strength")[cols].set_index("soc"), "Top 10 producers (out-strength)")
    log.table(nodes.nlargest(10, "in_strength")[cols].set_index("soc"), "Top 10 consumers (in-strength)")
    log.table(nodes.nlargest(10, "betweenness_w")[cols].set_index("soc"), "Top 10 brokers (weighted betweenness, distance = 1/w)")
    return stats, nodes


def top_edges(A, occs, titles, P, Cm, mult, art_names, k=15):
    idx = np.dstack(np.unravel_index(np.argsort(A, axis=None)[::-1][:k], A.shape))[0]
    csum, psum = Cm.values.sum(0), P.values.sum(0)
    g = 0.5 * (np.divide(1, csum, out=np.zeros_like(csum), where=csum > 0)
               + np.divide(1, psum, out=np.zeros_like(psum), where=psum > 0))
    rows = []
    for i, j in idx:
        contrib = P.values[i] * Cm.values[j] * g * mult[i, j]
        top = np.argsort(contrib)[::-1][:2]
        rows.append({"from": short_title(titles[occs[i]], 34), "to": short_title(titles[occs[j]], 34),
                     "flow": A[i, j], "share_of_total": A[i, j] / A.sum(),
                     "carried_by": "; ".join(f"{art_names[P.columns[a]]} ({contrib[a] / contrib.sum():.0%})"
                                            for a in top if contrib[a] >= 0.01 * contrib.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- clustering
def modularity(A, labels, directed=True, gamma=1.0):
    labels = np.asarray(labels)
    same = labels[:, None] == labels[None, :]
    if directed:
        m = A.sum()
        return float(((A - gamma * np.outer(A.sum(1), A.sum(0)) / m) * same).sum() / m)
    Bm = A + A.T
    m2 = Bm.sum()
    k = Bm.sum(1)
    return float(((Bm - gamma * np.outer(k, k) / m2) * same).sum() / m2)


def to_labels(comms, n):
    lab = np.empty(n, int)
    for c, members in enumerate(comms):
        lab[list(members)] = c
    return lab


def make_runner(algo, A, directed):
    """Return f(seed, resolution) -> label vector for a weighted adjacency matrix A."""
    n = len(A)
    if not directed:
        A = A + A.T if not np.allclose(A, A.T) else A
    if algo == "louvain":
        G = nx.from_numpy_array(A, create_using=nx.DiGraph if directed else nx.Graph)
        return lambda seed, res=1.0: to_labels(
            nx.community.louvain_communities(G, weight="weight", resolution=res, seed=seed), n)
    if algo == "leiden":
        g = ig.Graph.Weighted_Adjacency(A.tolist(), mode="directed" if directed else "max",
                                        attr="weight", loops=False)
        return lambda seed, res=1.0: np.array(la.find_partition(
            g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=res,
            seed=seed, n_iterations=-1).membership)
    raise ValueError(algo)


def coassignment(label_runs):
    L = np.asarray(label_runs)
    return (L[:, :, None] == L[:, None, :]).mean(0)


def consensus(algo, label_runs, n_runs, tau, max_iter=25):
    """Lancichinetti & Fortunato (2012) consensus clustering: threshold the co-assignment
    matrix at tau, re-cluster it with the same algorithm, iterate until all runs agree."""
    runs = label_runs
    for it in range(max_iter):
        D = coassignment(runs)
        np.fill_diagonal(D, 0)
        if np.all((D < 1e-12) | (D > 1 - 1e-12)):
            return canonical(runs[0]), it
        Dt = np.where(D >= tau, D, 0.0)
        for i in np.where((Dt.sum(1) == 0) & (D.sum(1) > 0))[0]:   # keep nodes attached to strongest partner
            j = int(np.argmax(D[i]))
            Dt[i, j] = Dt[j, i] = D[i, j]
        f = make_runner(algo, Dt, directed=False)
        runs = [f(s) for s in range(n_runs)]
    D = coassignment(runs)
    best = max(runs, key=lambda r: modularity(D, r, directed=False))
    return canonical(best), max_iter


def canonical(labels):
    """Relabel clusters 0..K-1 by decreasing size (ties by first member)."""
    labels = np.asarray(labels)
    order = sorted(set(labels), key=lambda c: (-(labels == c).sum(), np.where(labels == c)[0][0]))
    remap = {c: k for k, c in enumerate(order)}
    return np.array([remap[c] for c in labels])


def align(ref, other):
    """Relabel `other` to best match `ref` (Hungarian on the contingency table) for display."""
    ct = pd.crosstab(other, ref)
    r, c = linear_sum_assignment(-ct.values)
    remap = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
    nxt = max(ref.max(), max(remap.values(), default=-1)) + 1
    for o in ct.index:
        if o not in remap:
            remap[o] = nxt
            nxt += 1
    return np.array([remap[x] for x in other])


def n_substantive(labels):
    s = pd.Series(labels).value_counts()
    return int((s > 1).sum()), int((s == 1).sum())


def disconnected_communities(A, labels):
    """Number of communities (size > 1) whose induced subgraph is not weakly connected."""
    bad = 0
    for c in set(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            k, _ = connected_components((A[np.ix_(idx, idx)] > 0).astype(int), directed=True, connection="weak")
            bad += k > 1
    return bad


def ensemble(algo, A, n_seeds, tau, directed, log):
    t0 = time.time()
    f = make_runner(algo, A, directed)
    runs = [f(s) for s in range(n_seeds)]
    Qs = np.array([modularity(A, r, directed) for r in runs])
    Ks = np.array([n_substantive(r)[0] for r in runs])
    disc = np.array([disconnected_communities(A, r) for r in runs])
    pair_ari = np.array([ARI(runs[a], runs[b]) for a, b in combinations(range(n_seeds), 2)])
    cons, iters = consensus(algo, runs, n_seeds, tau)
    best = canonical(runs[int(np.argmax(Qs))])
    out = {"runs": runs, "Q": Qs, "K": Ks, "disconnected": disc, "pair_ari": pair_ari,
           "D": coassignment(runs), "consensus": cons, "consensus_iters": iters, "best": best,
           "Q_consensus": modularity(A, cons, directed), "Q_best": Qs.max(),
           "ARI_consensus_best": ARI(cons, best), "secs": time.time() - t0}
    k, iso = n_substantive(cons)
    log(f"  {algo:8s} {n_seeds} seeds: Q mean {Qs.mean():.4f} (sd {Qs.std():.4f}, range {Qs.min():.4f}-{Qs.max():.4f}); "
        f"clusters {Ks.min()}-{Ks.max()} (mode {pd.Series(Ks).mode()[0]}); pairwise ARI mean {pair_ari.mean():.3f}; "
        f"runs with a disconnected community {int((disc > 0).sum())}/{n_seeds}")
    log(f"  {'':8s} consensus (tau={tau}, {iters} re-clustering rounds): {k} clusters + {iso} singleton(s), "
        f"Q = {out['Q_consensus']:.4f}; ARI vs best single run {out['ARI_consensus_best']:.3f}  [{out['secs']:.1f}s]")
    return out


def resolution_sweep(A, directed, ref, gammas, n_seeds):
    rows = []
    for algo in ["louvain", "leiden"]:
        f = make_runner(algo, A, directed)
        for gm in gammas:
            runs = [f(s, gm) for s in range(n_seeds)]
            rows.append({"algo": algo, "gamma": gm,
                         "K_mean": np.mean([n_substantive(r)[0] for r in runs]),
                         "singletons_mean": np.mean([n_substantive(r)[1] for r in runs]),
                         "Q_mean": np.mean([modularity(A, r, directed) for r in runs]),
                         "ARI_vs_gamma1_consensus": np.mean([ARI(ref[algo], r) for r in runs]),
                         "pairwise_ARI": np.mean([ARI(runs[a], runs[b]) for a, b in combinations(range(n_seeds), 2)])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cluster profiles
def participation(A, labels):
    """Participation coefficient and within-module strength z-score (Guimera & Amaral 2005)
    on the symmetrised flow matrix."""
    S = A + A.T
    s = S.sum(1)
    K = labels.max() + 1
    s_c = np.stack([S[:, labels == c].sum(1) for c in range(K)], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pc = np.where(s > 0, 1 - ((s_c / s[:, None]) ** 2).sum(1), 0.0)
    own = s_c[np.arange(len(A)), labels]
    z = np.zeros(len(A))
    for c in range(K):
        m = labels == c
        if m.sum() > 1 and own[m].std() > 0:
            z[m] = (own[m] - own[m].mean()) / own[m].std()
    role = np.where(z >= 2.5,
                    np.select([Pc <= .30, Pc <= .75], ["R5 provincial hub", "R6 connector hub"], "R7 kinless hub"),
                    np.select([Pc <= .05, Pc <= .62, Pc <= .80],
                              ["R1 ultra-peripheral", "R2 peripheral", "R3 connector"], "R4 kinless"))
    return Pc, z, role


def cluster_profiles(A, labels, occs, titles, P, Cm, mult, art_names, art_domain, strength):
    csum, psum = Cm.values.sum(0), P.values.sum(0)
    g = 0.5 * (np.divide(1, csum, out=np.zeros_like(csum), where=csum > 0)
               + np.divide(1, psum, out=np.zeros_like(psum), where=psum > 0))
    total = A.sum()
    rows = []
    for c in range(labels.max() + 1):
        m = np.where(labels == c)[0]
        internal = A[np.ix_(m, m)].sum()
        out_f, in_f = A[m].sum() - internal, A[:, m].sum() - internal
        Mc = mult[np.ix_(m, m)]
        contrib = np.einsum("ia,ij,ja->a", P.values[m], Mc, Cm.values[m]) * g
        if internal > 0:
            assert abs(contrib.sum() - internal) < 1e-8 * max(1, internal)
        top = [a for a in np.argsort(contrib)[::-1][:3] if contrib[a] > 0]
        dom = pd.Series(contrib, index=[art_domain[a] for a in P.columns]).groupby(level=0).sum()
        members = sorted(m, key=lambda i: -strength[i])
        rows.append({"cluster": c + 1, "size": len(m),
                     "top_artifacts": ("; ".join(f"{art_names[P.columns[a]]} ({contrib[a] / internal:.0%})" for a in top)
                                       if internal > 0 else "(singleton: no internal flow)"),
                     "dominant_domain": dom.idxmax() if internal > 0 else "",
                     "internal_flow": internal, "out_flow": out_f, "in_flow": in_f,
                     "closure_out": internal / (internal + out_f) if internal + out_f > 0 else np.nan,
                     "closure_in": internal / (internal + in_f) if internal + in_f > 0 else np.nan,
                     "share_of_total_flow": internal / total,
                     "members_by_strength": " | ".join(short_title(titles[occs[i]], 40) for i in members)})
    return pd.DataFrame(rows).set_index("cluster")


def partition_metrics(A, labels):
    same = labels[:, None] == labels[None, :]
    flat = np.argsort(A, axis=None)[::-1][:100]
    i, j = np.unravel_index(flat, A.shape)
    return {"within_share": A[same].sum() / A.sum(), "top100_within": int(same[i, j].sum())}


# ------------------------------------------------------------------------- figures
def palette(K):
    """tab10 order without its grey (reserved for singletons), then the light tab20 shades."""
    t20 = plt.get_cmap("tab20").colors
    order = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19]
    return [t20[order[k % len(order)]] for k in range(K)]


def node_colors(labels, singletons):
    pal = palette(labels.max() + 1)
    return [(0.6, 0.6, 0.6) if l in singletons else pal[l] for l in labels]


def grouped_layout(A, lab, seed=7):
    """Clusters on a ring (ordered by a spring layout of the cluster-level flow graph, so that
    heavily linked clusters sit next to each other); members spring-laid inside each cluster."""
    S = A + A.T
    sizes = np.bincount(lab)
    multi = [c for c in range(len(sizes)) if sizes[c] > 1]
    C = np.array([[S[np.ix_(lab == a, lab == b)].sum() for b in multi] for a in multi])
    np.fill_diagonal(C, 0)
    pc = nx.spring_layout(nx.from_numpy_array(C / C.max()), weight="weight", seed=seed)
    order = sorted(range(len(multi)), key=lambda i: np.arctan2(pc[i][1], pc[i][0]))
    theta = {multi[i]: 2 * np.pi * k / len(multi) for k, i in enumerate(order)}
    rmax = 0.95 * np.sin(np.pi / max(len(multi), 3))
    pos, radius = {}, {}
    for c in multi:
        idx = np.where(lab == c)[0]
        center = np.array([np.cos(theta[c]), np.sin(theta[c])])
        radius[c] = rmax * np.sqrt(sizes[c] / sizes.max())
        Sc = S[np.ix_(idx, idx)]
        Sc = np.sqrt(Sc / Sc.max()) if Sc.max() > 0 else Sc     # damp strong pairs so members spread out
        p = nx.spring_layout(nx.from_numpy_array(Sc), weight="weight", seed=seed,
                             k=3.0 / np.sqrt(len(idx)), iterations=300)
        arr = np.array([p[k] for k in range(len(idx))])
        arr = (arr - arr.mean(0)) / max(1e-9, np.linalg.norm(arr - arr.mean(0), axis=1).max())
        for k, i in enumerate(idx):
            pos[i] = center + radius[c] * arr[k]
    singles = [i for i in range(len(A)) if sizes[lab[i]] == 1]
    for t, i in enumerate(singles):
        pos[i] = np.array([-1.3 + 0.25 * t, -1.3])
    return pos, theta, radius


def fig_network(A, backbone, labs, occs, titles, strength, out, spec_label, directed=True):
    ref = labs["Louvain consensus"]
    pos, theta, radius = grouped_layout(A, ref)
    fig, axes = plt.subplots(1, 2, figsize=(21, 10.5))
    G = nx.from_numpy_array(np.where(backbone, A, 0), create_using=nx.DiGraph)
    E = list(G.edges())
    ew = np.array([G[u][v]["weight"] for u, v in E])
    size = 20 + 520 * np.sqrt(strength / strength.max())
    for ax, (name, lab) in zip(axes, labs.items()):
        sing = {c for c, n in pd.Series(lab).value_counts().items() if n == 1}
        cols = node_colors(lab, sing)
        intra = np.array([lab[u] == lab[v] for u, v in E])
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[e for e, k in zip(E, intra) if not k],
                               width=0.2 + 2.0 * ew[~intra] / ew.max(), alpha=0.10, edge_color="0.35", arrows=False)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[e for e, k in zip(E, intra) if k],
                               width=0.3 + 2.5 * ew[intra] / ew.max(), alpha=0.45,
                               edge_color=[cols[u] for (u, v), k in zip(E, intra) if k], arrows=False)
        diff = np.where(lab != ref)[0]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=size, node_color=cols, edgecolors="white", linewidths=0.6)
        if name != "Louvain consensus" and len(diff):
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(diff), node_size=size[diff] * 1.6,
                                   node_color="none", edgecolors="black", linewidths=1.6)
        labelled = set(diff) if name != "Louvain consensus" else set()
        for c in sorted(set(ref)):       # three strongest members of each (Louvain) group
            m = np.where(ref == c)[0]
            labelled |= set(m[np.argsort(-strength[m])][:2])
        for i in labelled:
            ax.annotate(short_title(titles[occs[i]], 26), pos[i], fontsize=6.3, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
        for c, th in theta.items():    # group labels (Louvain numbering) outside the ring
            r = 1 + radius[c] + 0.08
            ax.text(r * np.cos(th), r * np.sin(th), f"C{c + 1}", fontsize=12, fontweight="bold",
                    ha="center", va="center", color=palette(ref.max() + 1)[c])
        Q = modularity(A, lab, directed)
        k, iso = n_substantive(lab)
        extra = f"\nringed: {len(diff)} job(s) assigned differently from Louvain" if name != "Louvain consensus" else "\n"
        ax.set_title(f"{name}: {k} clusters + {iso} singleton(s), Q = {Q:.3f}{extra}", fontsize=12)
        handles = [Line2D([], [], marker="o", ls="", color=palette(lab.max() + 1)[c],
                          label=f"C{c + 1} (n={int((lab == c).sum())})") for c in sorted(set(lab)) if c not in sing]
        handles += [Line2D([], [], marker="o", ls="", color="0.6",
                           label="singleton: " + ", ".join(short_title(titles[occs[i]], 20)
                                                           for i in np.where(np.isin(lab, list(sing)))[0]))] if sing else []
        ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False, ncol=2)
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
    kept = A[backbone].sum() / A.sum()
    fig.suptitle(f"Handoff network, {spec_label}. Edges: disparity backbone (a = .05; {int(backbone.sum())} edges, "
                 f"{kept:.0%} of flow); coloured = within cluster, grey = between. Node size ~ sqrt(in + out strength).\n"
                 f"Both panels use the same positions, grouped by the Louvain consensus.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_blocks(A, labs, out, spec_label):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), layout="constrained")
    pos = A[A > 0]
    norm = LogNorm(vmin=np.percentile(pos, 5), vmax=pos.max())
    for ax, (name, lab) in zip(axes, labs.items()):
        strength = A.sum(0) + A.sum(1)
        order = np.lexsort((-strength, lab))
        M = np.where(A[np.ix_(order, order)] > 0, A[np.ix_(order, order)], np.nan)
        im = ax.imshow(M, cmap="magma_r", norm=norm, interpolation="nearest")
        bounds = np.cumsum(pd.Series(lab).value_counts().sort_index().values)[:-1] - 0.5
        for b in bounds:
            ax.axhline(b, color="tab:blue", lw=0.7)
            ax.axvline(b, color="tab:blue", lw=0.7)
        ticks = np.r_[0, bounds + 0.5]
        sizes = pd.Series(lab).value_counts().sort_index().values
        ax.set_xticks(ticks + sizes / 2 - 0.5)
        ax.set_xticklabels([f"C{c + 1}" for c in range(len(sizes))], fontsize=7, rotation=90)
        ax.set_yticks(ticks + sizes / 2 - 0.5)
        ax.set_yticklabels([f"C{c + 1}" for c in range(len(sizes))], fontsize=7)
        m = partition_metrics(A, lab)
        ax.set_title(f"{name}: {m['within_share']:.0%} of flow within clusters", fontsize=11)
        ax.set_xlabel("consumer (to)")
        ax.set_ylabel("producer (from)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="flow F[i,j] (log scale)")
    fig.suptitle(f"Flow matrix ordered by cluster (members by strength within cluster): {spec_label}", fontsize=12)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_contingency(lou, lei, out, spec_label):
    ct = pd.crosstab(pd.Series(lou, name="Louvain"), pd.Series(lei, name="Leiden"))
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * ct.shape[1], 1.2 + 0.5 * ct.shape[0]))
    ax.imshow(ct.values, cmap="Blues")
    for (i, j), v in np.ndenumerate(ct.values):
        if v:
            ax.text(j, i, v, ha="center", va="center", fontsize=9,
                    color="white" if v > ct.values.max() / 2 else "black")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels([f"C{c + 1}" for c in ct.columns], fontsize=8)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels([f"C{c + 1}" for c in ct.index], fontsize=8)
    ax.set_xlabel("Leiden consensus (aligned)")
    ax.set_ylabel("Louvain consensus")
    ax.set_title(f"{spec_label}\nARI = {ARI(lou, lei):.3f}, NMI = {NMI(lou, lei):.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_ensembles(res, lou, out, spec_label):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    c = {"louvain": "tab:orange", "leiden": "tab:green"}
    ax = axes[0, 0]
    for a in res:
        ax.hist(res[a]["Q"], bins=20, alpha=0.6, color=c[a], label=f"{a} (max {res[a]['Q'].max():.4f})")
        ax.axvline(res[a]["Q_consensus"], color=c[a], ls="--", lw=1.2)
    ax.set_xlabel("directed modularity Q (per seed); dashed = consensus partition")
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ks = sorted(set(np.r_[res["louvain"]["K"], res["leiden"]["K"]]))
    wdt = 0.4
    for t, a in enumerate(res):
        cnt = pd.Series(res[a]["K"]).value_counts().reindex(ks, fill_value=0)
        ax.bar(np.array(ks) + (t - 0.5) * wdt, cnt.values, width=wdt, color=c[a], label=a)
    ax.set_xticks(ks)
    ax.set_xlabel("clusters of size > 1 (per seed)")
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for a in res:
        ax.hist(res[a]["pair_ari"], bins=30, alpha=0.6, color=c[a],
                label=f"{a} (mean {res[a]['pair_ari'].mean():.3f})")
    ax.set_xlabel("ARI between pairs of seeds")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    order = np.argsort(lou, kind="stable")
    im = ax.imshow(res["louvain"]["D"][np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
    for b in np.cumsum(pd.Series(lou).value_counts().sort_index().values)[:-1] - 0.5:
        ax.axhline(b, color="white", lw=0.5)
        ax.axvline(b, color="white", lw=0.5)
    ax.set_title("Louvain co-assignment across seeds (ordered by consensus)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of runs together")
    fig.suptitle(f"Seed ensembles: {spec_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_descriptives(A, nodes, lab, Pc, z, out, spec_label):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sing = {c for c, n in pd.Series(lab).value_counts().items() if n == 1}
    cols = node_colors(lab, sing)
    ax = axes[0, 0]
    ok = (nodes.out_strength > 0) & (nodes.in_strength > 0)
    ax.scatter(nodes.out_strength[ok], nodes.in_strength[ok], s=18, c=np.array(cols)[ok])
    lim = [nodes.loc[ok, ["out_strength", "in_strength"]].min().min() * 0.8,
           nodes[["out_strength", "in_strength"]].max().max() * 1.2]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("out-strength (produced handoffs)")
    ax.set_ylabel("in-strength (consumed handoffs)")
    offs = [(5, 4), (5, -9), (-5, 6), (-5, -9), (5, 10), (5, -15)]
    for k, i in enumerate(nodes.net_producer.abs().nlargest(6).index):
        ax.annotate(short_title(nodes.title[i], 26), (nodes.out_strength[i], nodes.in_strength[i]), fontsize=6.5,
                    xytext=offs[k], textcoords="offset points", ha="left" if offs[k][0] > 0 else "right")
    ax.set_title("Producers vs consumers (colour = Louvain consensus)", fontsize=10)
    ax = axes[0, 1]
    w = np.sort(A[A > 0])
    ax.loglog(w, 1 - np.arange(len(w)) / len(w), ".", ms=2)
    ax.set_xlabel("edge weight w")
    ax.set_ylabel("P(W >= w)")
    ax.set_title(f"Edge-weight CCDF (Gini {gini(w):.2f})", fontsize=10)
    ax = axes[1, 0]
    ts = nodes.total_strength
    ax.hist(ts[ts > 0], bins=np.logspace(np.log10(ts[ts > 0].min()), np.log10(ts.max()), 30), color="tab:blue", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("total strength (in + out)")
    ax.set_ylabel("occupations")
    ax.set_title("Strength distribution", fontsize=10)
    ax = axes[1, 1]
    keep = ~np.isin(lab, list(sing))
    ax.scatter(Pc[keep], z[keep], s=18, c=np.array(cols)[keep])
    for x in [0.05, 0.62, 0.80]:
        ax.axvline(x, color="0.7", lw=0.7, ls="--")
    ax.axhline(2.5, color="0.7", lw=0.7, ls="--")
    top = [i for i in np.argsort(-Pc) if keep[i]][:6]
    for k, i in enumerate(sorted(top, key=lambda i: z[i])):
        ax.annotate(short_title(nodes.title[i], 26), (Pc[i], z[i]), fontsize=6.5, xytext=(-8, 4 + 3 * (k % 2)),
                    textcoords="offset points", ha="right",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
    ax.set_xlabel("participation coefficient P")
    ax.set_ylabel("within-cluster strength z")
    ax.set_title("Guimera-Amaral roles (Louvain consensus): boundary jobs to the right", fontsize=10)
    fig.suptitle(f"Descriptive distributions: {spec_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_cluster_flows(A, labs, out, spec_label):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), layout="constrained")
    for ax, (name, lab) in zip(axes, labs.items()):
        K = lab.max() + 1
        C = np.array([[A[np.ix_(lab == a, lab == b)].sum() for b in range(K)] for a in range(K)])
        R = C / np.where(C.sum(1, keepdims=True) > 0, C.sum(1, keepdims=True), 1)
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(R):
            if v >= 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"C{c + 1}" for c in range(K)], fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"C{c + 1} (n={int((lab == c).sum())})" for c in range(K)], fontsize=8)
        ax.set_xlabel("to cluster")
        ax.set_ylabel("from cluster")
        ax.set_title(f"{name}: share of each cluster's out-flow by destination", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"Inter-cluster handoffs: {spec_label}", fontsize=12)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_sweep(sw, out, spec_label):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    c = {"louvain": "tab:orange", "leiden": "tab:green"}
    for a, d in sw.groupby("algo"):
        axes[0].plot(d.gamma, d.K_mean, "o-", color=c[a], label=a)
        axes[1].plot(d.gamma, d.Q_mean, "o-", color=c[a], label=a)
        axes[2].plot(d.gamma, d.ARI_vs_gamma1_consensus, "o-", color=c[a], label=f"{a}: vs gamma=1 consensus")
        axes[2].plot(d.gamma, d.pairwise_ARI, "s:", color=c[a], label=f"{a}: between seeds")
    for ax in axes:
        ax.axvline(1.0, color="0.7", ls="--", lw=0.8)
        ax.set_xlabel("resolution gamma")
    axes[0].set_ylabel("mean clusters (size > 1)")
    axes[1].set_ylabel("mean modularity (gamma = 1 objective)")
    axes[2].set_ylabel("mean ARI")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"Resolution sweep: {spec_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def analyse_spec(name, spec, occs, titles, P, Cm, art_names, art_domain, args, log, out_dir):
    A, mult, label = spec["A"], spec["mult"], spec["label"]
    os.makedirs(out_dir, exist_ok=True)
    log.section(f"SPECIFICATION: {label}")
    backbone = disparity_backbone(A, args.alpha)
    stats, nodes = describe(A, occs, titles, backbone, log, label)
    te = top_edges(A, occs, titles, P, Cm, mult, art_names)
    log.table(te, "Top 15 handoffs (directed) and the artifacts carrying them")

    log(f"\nCommunity detection ({'directed' if args.directed else 'symmetrised'} modularity, "
        f"resolution {args.resolution}):")
    res = {a: ensemble(a, A, args.n_seeds, args.tau, args.directed, log) for a in ["louvain", "leiden"]}
    lou = res["louvain"]["consensus"]
    lei = align(lou, res["leiden"]["consensus"])
    labs = {"Louvain consensus": lou, "Leiden consensus": lei}

    cmp_rows = []
    for a, lab in [("Louvain", lou), ("Leiden", lei)]:
        k, iso = n_substantive(lab)
        pm = partition_metrics(A, lab)
        cmp_rows.append({"algorithm": a, "clusters": k, "singletons": iso,
                         "sizes": "-".join(map(str, sorted(pd.Series(lab).value_counts(), reverse=True))),
                         "Q_consensus": modularity(A, lab, args.directed),
                         "Q_best_seed": res[a.lower()]["Q_best"],
                         "Q_mean_seed": res[a.lower()]["Q"].mean(),
                         "pairwise_ARI_seeds": res[a.lower()]["pair_ari"].mean(),
                         "runs_with_disconnected_comm": int((res[a.lower()]["disconnected"] > 0).sum()),
                         "within_flow_share": pm["within_share"], "top100_within": pm["top100_within"]})
    cmp = pd.DataFrame(cmp_rows).set_index("algorithm")
    log.table(cmp, "Louvain vs Leiden: consensus partitions")
    log(f"Agreement between consensus partitions: ARI = {ARI(lou, lei):.3f}, NMI = {NMI(lou, lei):.3f}")
    moved = np.where(lou != lei)[0]
    if len(moved):
        mv = pd.DataFrame({"soc": [occs[i] for i in moved], "title": [titles[occs[i]] for i in moved],
                           "louvain": [f"C{lou[i] + 1}" for i in moved], "leiden": [f"C{lei[i] + 1}" for i in moved]})
        log.table(mv.set_index("soc"), f"Occupations assigned differently ({len(moved)})")

    strength = nodes.total_strength.values
    for a, lab in [("louvain", lou), ("leiden", lei)]:
        prof = cluster_profiles(A, lab, occs, titles, P, Cm, mult, art_names, art_domain, strength)
        prof.to_csv(os.path.join(out_dir, f"clusters_{a}.csv"))
        show = prof.drop(columns=["members_by_strength"]).copy()
        log.table(show, f"Cluster profiles: {a} consensus (closure_out = share of the cluster's produced flow consumed inside it)")
        log(f"\nMembers ({a}, by total strength):")
        for c, r in prof.iterrows():
            log(f"  C{c} (n={r['size']}): {r['members_by_strength']}")

    Pc, z, role = participation(A, lou)
    Pc_l, z_l, role_l = participation(A, lei)
    nodes["louvain"], nodes["leiden"] = lou + 1, lei + 1
    nodes["participation_louvain"], nodes["z_louvain"], nodes["role_louvain"] = Pc, z, role
    nodes["participation_leiden"], nodes["z_leiden"], nodes["role_leiden"] = Pc_l, z_l, role_l
    nodes["coassign_stability_louvain"] = [res["louvain"]["D"][i, lou == lou[i]].mean() for i in range(len(occs))]
    nodes.to_csv(os.path.join(out_dir, "nodes.csv"), index=False)
    log.table(nodes.nlargest(15, "participation_louvain")[["soc", "title", "louvain", "leiden",
                                                            "participation_louvain", "z_louvain", "role_louvain"]]
              .set_index("soc"), "Boundary jobs: top 15 participation coefficients (Louvain consensus)")
    log.table(pd.Series(role).value_counts().rename("count").to_frame(), "Guimera-Amaral roles (Louvain consensus)")

    htp = pd.DataFrame({"job": list(HIRE_TO_PAY.values()), "title": [titles[s] for s in HIRE_TO_PAY],
                        "louvain": [f"C{lou[occs.index(s)] + 1}" for s in HIRE_TO_PAY],
                        "leiden": [f"C{lei[occs.index(s)] + 1}" for s in HIRE_TO_PAY]},
                       index=pd.Index(list(HIRE_TO_PAY), name="soc"))
    log.table(htp, "Validation: hire-to-pay jobs (stage 1)")

    sw = None
    if not args.no_sweep:
        gammas = np.round(np.arange(args.sweep_min, args.sweep_max + 1e-9, args.sweep_step), 3)
        sw = resolution_sweep(A, args.directed, {"louvain": lou, "leiden": res["leiden"]["consensus"]},
                              gammas, args.sweep_seeds)
        sw.to_csv(os.path.join(out_dir, "resolution_sweep.csv"), index=False)
        log.table(sw.set_index(["algo", "gamma"]), f"Resolution sweep ({args.sweep_seeds} seeds per gamma)")

    # tables
    stats.to_csv(os.path.join(out_dir, "network_stats.csv"))
    cmp.to_csv(os.path.join(out_dir, "louvain_vs_leiden.csv"))
    te.to_csv(os.path.join(out_dir, "top_edges.csv"), index=False)
    i, j = np.where(A > 0)
    pd.DataFrame({"from": [occs[k] for k in i], "to": [occs[k] for k in j], "flow": A[i, j],
                  "backbone": backbone[i, j], "same_cluster_louvain": lou[i] == lou[j],
                  "same_cluster_leiden": lei[i] == lei[j]}).to_csv(os.path.join(out_dir, "edges.csv"), index=False)
    pd.DataFrame([{"algo": a, "seed": s, "Q": res[a]["Q"][s], "clusters": res[a]["K"][s],
                   "disconnected_communities": res[a]["disconnected"][s]}
                  for a in res for s in range(args.n_seeds)]).to_csv(os.path.join(out_dir, "seed_runs.csv"), index=False)

    # figures
    fig_network(A, backbone, labs, occs, titles, strength, os.path.join(out_dir, "fig1_network.png"), label,
                args.directed)
    fig_blocks(A, labs, os.path.join(out_dir, "fig2_flow_matrix_blocks.png"), label)
    fig_contingency(lou, lei, os.path.join(out_dir, "fig3_louvain_vs_leiden.png"), label)
    fig_ensembles(res, lou, os.path.join(out_dir, "fig4_seed_ensembles.png"), label)
    fig_descriptives(A, nodes, lou, Pc, z, os.path.join(out_dir, "fig5_descriptives.png"), label)
    fig_cluster_flows(A, labs, os.path.join(out_dir, "fig6_intercluster_flows.png"), label)
    if sw is not None:
        fig_sweep(sw, os.path.join(out_dir, "fig7_resolution_sweep.png"), label)
    return {"louvain": lou, "leiden": lei, "Q": {a: modularity(A, labs[k], args.directed)
                                                 for a, k in [("louvain", "Louvain consensus"), ("leiden", "Leiden consensus")]},
            "metrics": {a: partition_metrics(A, l) for a, l in [("louvain", lou), ("leiden", lei)]},
            "Q_best": {a: res[a]["Q_best"] for a in res}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/mnt/project")
    ap.add_argument("--code-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder containing network.py and copresence.py (claude/handoff)")
    ap.add_argument("--cache-dir", default=None, help="scratch dir for network.py inputs (default: network.BASE)")
    ap.add_argument("--out-dir", default="handoff_communities_out")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5, help="consensus threshold")
    ap.add_argument("--undirected", dest="directed", action="store_false",
                    help="cluster the symmetrised flow matrix F + F' instead of the directed one")
    ap.add_argument("--alpha", type=float, default=0.05, help="disparity-filter significance for the backbone")
    ap.add_argument("--impute", choices=["median", "mean", "equal"], default="median",
                    help="weight for tasks without O*NET ratings (equal = 1, before normalising)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--sweep-min", type=float, default=0.25)
    ap.add_argument("--sweep-max", type=float, default=2.5)
    ap.add_argument("--sweep-step", type=float, default=0.25)
    ap.add_argument("--sweep-seeds", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true", help="rebuild cached inputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = Reporter(os.path.join(args.out_dir, "summary.txt"))
    dirs = [args.data_dir, args.code_dir]
    N, npath = import_module("network", [args.code_dir, args.data_dir])
    CP, cpath = import_module("copresence", [args.code_dir, args.data_dir])
    cache = args.cache_dir or N.BASE
    N.BASE = os.path.join(cache, "")
    files = {"codes": find_file(dirs, variants("final_codes.csv")),
             "vocab": find_file(dirs, variants("artifact_vocabulary.csv")),
             "tasks": find_file(dirs, variants("Task Statements.xlsx")),
             "ratings": find_file(dirs, variants("Task Ratings.xlsx")),
             "oews": find_file(dirs, ["nat4d_M2025_dl.xlsx"])}
    log(f"network.py: {npath}\ncopresence.py: {cpath}")
    log("Inputs:", json.dumps(files, indent=1))
    build_cache(files, cache, args.impute, args.rebuild, log)

    specs, occs, titles, P, Cm, art_names, art_domain, info = build_networks(N, CP, cache, log)
    log(f"\nPilot: {info['n_tasks']} tasks, {info['n_occ']} occupations, {info['n_artifacts']} artifacts in the "
        f"main-spec P/C matrices; {info['n_imputed']} tasks with imputed weights.")
    log(f"Co-presence filter: {info['pairs_discounted']:.1%} of occupation pairs discounted (W < 1); "
        f"OEWS broad-code fallback for {len(info['broad_matches'])} occupations: {', '.join(info['broad_matches'])}; "
        f"special rule: {info['special_matches']}")

    results = {}
    for name in ["filtered", "unfiltered"]:
        results[name] = analyse_spec(name, specs[name], occs, titles, P, Cm, art_names, art_domain, args, log,
                                     os.path.join(args.out_dir, name))

    # ---- reproduction check against handoff_network_notes.md
    log.section("REPRODUCTION CHECK against handoff_network_notes.md (the notes report Louvain, resolution 1.0)")

    def htp_ok(lab):
        core = {lab[occs.index(s)] for s in HTP_CORE}
        acc = lab[occs.index("13-2011.00")]
        return "yes" if len(core) == 1 and acc not in core else f"no ({len(core)} clusters)"

    def moved(a, b):   # occupations whose (aligned) cluster differs between two partitions
        return int((align(a, b) != a).sum())

    rows = []
    for spec in ["unfiltered", "filtered"]:
        rep, r = REPORTED[spec], results[spec]
        for metric, reported, fn in [
            ("modularity Q (consensus)", rep["Q"], lambda a: r["Q"][a]),
            ("modularity Q (best of seeds)", None, lambda a: r["Q_best"][a]),
            ("share of flow within clusters", rep["within_share"], lambda a: r["metrics"][a]["within_share"]),
            ("top-100 handoffs within a cluster", rep.get("top100_within"), lambda a: r["metrics"][a]["top100_within"]),
            ("clusters (size > 1) + singletons",
             None if rep["n_clusters"] is None else f"{rep['n_clusters']} + {rep['n_isolates']}",
             lambda a: "{} + {}".format(*n_substantive(r[a]))),
            ("cluster sizes", REPORTED.get(f"sizes_{spec}"),
             lambda a: "-".join(map(str, sorted(pd.Series(r[a]).value_counts(), reverse=True)))),
            ("hire-to-pay: 5 core jobs together, ACC outside", "yes", lambda a: htp_ok(r[a])),
        ]:
            rows.append({"spec": spec, "metric": metric, "reported (notes)": reported,
                         "louvain": fn("louvain"), "leiden": fn("leiden")})
    for metric, reported, fn in [
        ("ARI filtered vs unfiltered", REPORTED["ARI_filtered_vs_unfiltered"],
         lambda a: ARI(results["filtered"][a], results["unfiltered"][a])),
        ("occupations moved by the filter", REPORTED["moved_by_filter"],
         lambda a: moved(results["unfiltered"][a], results["filtered"][a])),
    ]:
        rows.append({"spec": "both", "metric": metric, "reported (notes)": reported,
                     "louvain": fn("louvain"), "leiden": fn("leiden")})
    rows.append({"spec": "filter", "metric": "share of pairs discounted (W < 1)",
                 "reported (notes)": REPORTED["pairs_discounted"], "louvain": info["pairs_discounted"], "leiden": ""})
    chk = pd.DataFrame(rows)
    chk.to_csv(os.path.join(args.out_dir, "reproduction_check.csv"), index=False)
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
    log(chk.map(fmt).to_string(index=False))
    log(f"\nOutputs written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
