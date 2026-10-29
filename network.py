"""Producer-consumer handoff network for SOC 11/13/43 occupations (pilot).
 
Flow from occupation i to occupation j through artifact a:
  P[i,a] = time share of i spent producing a for internal recipients (task weight split equally across its outputs)
  C[j,a] = time share of j spent consuming a from internal suppliers (task weight split equally across its inputs)
  producer-allocated  F_P(i->j) = sum_a P[i,a] * C[j,a] / sum_k C[k,a]
  consumer-allocated  F_C(i->j) = sum_a C[j,a] * P[i,a] / sum_k P[k,a]
  F = (F_P + F_C) / 2, self-loops removed. Generic, relational and background artifacts excluded in the main spec.
Optional co-employment weight E[i,j] (from BLS industry staffing patterns) multiplies F when supplied.
"""
import json, re
import numpy as np, pandas as pd
 
BASE = ""
RELATIONAL = {"A333", "A204"}          # supervision / directives: hierarchy, not handoffs
BACKGROUND = {"A116", "A340", "A257"}  # law, external news, market data: no internal producer in scope
 
NEW_SYNONYMS = {"legal pleading / court filing": "court filing / pleading",
                "court filing / pleading": "court filing / pleading"}
 
 
def load():
    p = pd.read_pickle(BASE + "pilot_tasks.pkl").set_index("Task ID")
    codes = {t["task_id"]: t for t in json.load(open(BASE + "final_codes.json"))}
    voc = pd.DataFrame(json.load(open(BASE + "artifact_vocabulary.json"))).set_index("id")
    return p, codes, voc
 
 
def art_id(e):
    if e["artifact_id"] != "NEW":
        return e["artifact_id"]
    n = re.sub(r"\s+", " ", e.get("new_name", "").strip().lower())
    return "NEW:" + NEW_SYNONYMS.get(n, n)
 
 
def matrices(p, codes, voc, exclude_generic=True, exclude_relational=True, exclude_background=True,
             internal_only=True):
    generic = set(voc.index[voc.generic])
    drop = set()
    if exclude_generic: drop |= generic
    if exclude_relational: drop |= RELATIONAL
    if exclude_background: drop |= BACKGROUND
    prow, crow = [], []
    for tid, r in p.iterrows():
        t = codes[tid]
        outs = [art_id(e) for e in t["outputs"] if (not internal_only or e["party"] in ("internal", "either"))]
        ins = [art_id(e) for e in t["inputs"] if (not internal_only or e["party"] in ("internal", "either"))]
        # weight split over ALL listed outputs/inputs (so dropping an artifact does not inflate the others)
        n_o = max(1, len(t["outputs"])); n_i = max(1, len(t["inputs"]))
        for a in outs:
            if a not in drop:
                prow.append((r["O*NET-SOC Code"], a, r.w / n_o))
        for a in ins:
            if a not in drop:
                crow.append((r["O*NET-SOC Code"], a, r.w / n_i))
    P = pd.DataFrame(prow, columns=["occ", "art", "v"]).pivot_table(index="occ", columns="art", values="v", aggfunc="sum").fillna(0)
    Cm = pd.DataFrame(crow, columns=["occ", "art", "v"]).pivot_table(index="occ", columns="art", values="v", aggfunc="sum").fillna(0)
    occs = sorted(p["O*NET-SOC Code"].unique())
    arts = sorted(set(P.columns) | set(Cm.columns))
    P = P.reindex(index=occs, columns=arts, fill_value=0.0)
    Cm = Cm.reindex(index=occs, columns=arts, fill_value=0.0)
    return P, Cm
 
 
def flows(P, Cm, E=None):
    Pv, Cv = P.values, Cm.values
    csum = Cv.sum(0); psum = Pv.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cn = np.where(csum > 0, Cv / csum, 0.0)
        Pn = np.where(psum > 0, Pv / psum, 0.0)
    FP = Pv @ Cn.T
    FC = Pn @ Cv.T
    F = (FP + FC) / 2
    np.fill_diagonal(F, 0.0)
    if E is not None:
        F = F * E
    return pd.DataFrame(F, index=P.index, columns=P.index)
 
 
def artifact_contrib(P, Cm, i, j):
    """Which artifacts carry the flow i -> j."""
    csum = Cm.sum(0); psum = P.sum(0)
    fp = P.loc[i] * Cm.loc[j] / csum.replace(0, np.nan)
    fc = Cm.loc[j] * P.loc[i] / psum.replace(0, np.nan)
    return ((fp.fillna(0) + fc.fillna(0)) / 2).sort_values(ascending=False)