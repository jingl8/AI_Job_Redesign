import pandas as pd, numpy as np
def industry_matrix(occs):
    df=pd.read_pickle('/home/claude/handoff/oews4d.pkl'); df['emp']=pd.to_numeric(df.TOT_EMP,errors='coerce').fillna(0)
    tot=df[df.OCC_CODE=='00-0000'].set_index('NAICS').emp
    rows={}
    for o in occs:
        soc=o[:7]
        if soc=='11-1031':   # Legislators: not published by industry; place in state and local government
            v=tot[tot.index.astype(str).isin(['999200','999300'])]
            rows[o]=('state/local govt',v); continue
        cands=[soc, soc[:6]+'0']  # detailed, then broad
        for c in cands:
            sub=df[df.OCC_CODE==c]
            if len(sub): rows[o]=(c,sub.groupby('NAICS').emp.sum()); break
    M=pd.DataFrame({o:v for o,(c,v) in rows.items()}).T.reindex(columns=tot.index).fillna(0)
    used={o:c for o,(c,v) in rows.items()}
    return M, tot, used
def copresence(M, tot):
    S=M.div(M.sum(1),axis=0)                         # industry shares of each occupation
    share_k=tot/tot.sum()
    LQ=(M.div(tot,axis=1)).div(M.sum(1)/tot.sum(),axis=0)  # location quotient of occ j in industry k
    E=S.values @ LQ.values.T                          # E[i,j] = sum_k s_ik LQ_jk
    E=pd.DataFrame(E,index=M.index,columns=M.index)
    Esym=np.sqrt(E*E.T)
    W=np.minimum(1.0,Esym)
    weitz=pd.DataFrame(np.array([[np.minimum(S.values[a],S.values[b]).sum() for b in range(len(S))] for a in range(len(S))]),index=M.index,columns=M.index)
    return E, Esym, W, weitz, S