import numpy as np
import pandas as pd
from scipy import stats

N_VARS = 7
L      = 1
VAR_NAMES  = ["ln_real_gdp","ln_pce_goods","tightness",
               "ln_import_price","spare_capacity","fed_funds","acr_proxy"]

THRESHOLD  = 3.34

def sign_restrictions():
    return {0:0, 1:0, 2:0, 3:0, 4:0, 5:+1, 6:0}

def penalty_function(impact, signs):
    for idx, s in signs.items():
        if s != 0 and s * impact[idx] < 0:
            return abs(impact[idx])
    return 0.0

def identify_mp_shock(B, Sigma, signs, n_rot=5000):
    try:
        chol = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None
    for _ in range(n_rot):
        Q, _ = np.linalg.qr(np.random.randn(N_VARS, N_VARS))
        for col in range(N_VARS):
            impact = chol @ Q[:, col]
            if penalty_function(impact, signs) == 0.0:
                return impact
    return None

def build_var_matrices(data, mask, L=1):
    T, n = data.shape
    rows = [t for t in range(L, T) if mask[t]]
    if not rows:
        return np.empty((0,n)), np.empty((0, n*L+1))
    Y = data[rows]
    X = np.column_stack([data[np.array(rows)-l] for l in range(1,L+1)]
                        + [np.ones(len(rows))])
    return Y, X

def gibbs_var(Y, X, n_draws=500, burnin=200, lam=0.1):
    T_r, n = Y.shape
    k = X.shape[1]
    if T_r < n + 2:
        return [], []
    XtX = X.T @ X
    XtY = X.T @ Y
    B_ols = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
    resid = Y - X @ B_ols
    S_ols = resid.T @ resid / max(T_r-k,1) + 1e-4*np.eye(n)
    V0_inv = np.eye(k) / (lam**2)
    B0     = np.zeros((k,n))
    Sigma  = S_ols.copy()
    B_out, S_out = [], []
    for draw in range(n_draws + burnin):
        Vn = np.linalg.inv(XtX + V0_inv + 1e-8*np.eye(k))
        Bn = Vn @ (XtY + V0_inv @ B0)
        try:
            B = Bn + np.linalg.cholesky(Vn) @ np.random.randn(k,n)
        except np.linalg.LinAlgError:
            B = Bn
        res_b = Y - X @ B
        Sn    = S_ols + res_b.T @ res_b
        Sn    = 0.5*(Sn+Sn.T) + 1e-6*np.eye(n)
        try:
            Sigma = stats.invwishart.rvs(df=T_r+n+2, scale=Sn)
        except Exception:
            Sigma = Sn/(T_r+n+2)
        if draw >= burnin:
            B_out.append(B); S_out.append(Sigma)
    return B_out, S_out

def estimate_tvar(data_df, threshold_var="acr_proxy", threshold=THRESHOLD,
                  n_draws=500, burnin=200, n_burn=None, lam=0.1, h_irf=24):
    if n_burn is not None:
        burnin = n_burn
    cols  = VAR_NAMES
    df    = data_df[cols].dropna()
    data  = df.values
    T     = len(data)
    gscpi = df[threshold_var].values
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        if gscpi[t-1] > threshold:
            regime[t] = 1
    print(f"Threshold: {threshold:.2f}")
    print(f"Disrupted months (regime=1): {regime.sum()}")
    print(f"Undisrupted months (regime=0): {(regime==0).sum()}")
    signs   = sign_restrictions()
    results = {}
    for reg in [0,1]:
        label = "disrupted" if reg==1 else "undisrupted"
        Y, X  = build_var_matrices(data, regime==reg)
        print(f"\nRegime {reg} ({label}): {Y.shape[0]} obs")
        B_draws, S_draws = gibbs_var(Y, X, n_draws=n_draws, burnin=burnin, lam=lam)
        print(f"  Posterior draws: {len(B_draws)}")
        irf_draws = []
        for B, Sigma in zip(B_draws, S_draws):
            impact = identify_mp_shock(B, Sigma, signs)
            if impact is not None:
                irf = [impact.copy()]
                Bc  = B[:N_VARS*L, :].T
                st  = impact.copy()
                for _ in range(h_irf-1):
                    st = Bc @ st
                    irf.append(st.copy())
                irf_draws.append(np.array(irf))
        print(f"  Valid IRF draws: {len(irf_draws)}")
        results[label] = np.array(irf_draws) if irf_draws else None
    return results, regime

def save_results(results, regime=None, output_dir="results"):
    import pickle, os
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,"tvar_results.pkl"),"wb") as fh:
        pickle.dump({"results":results,"regime":regime}, fh)
    print(f"Results saved to {output_dir}/tvar_results.pkl")
