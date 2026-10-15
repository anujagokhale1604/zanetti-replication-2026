import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from zanetti.tvar_estimation import (
    VAR_NAMES, ENDOG_VARS, N_VARS, THRESHOLD,
    sign_restrictions, identify_mp_shock
)

def extract_var_residuals(data_df, results):
    df    = data_df[VAR_NAMES].dropna()
    data  = df.values
    T     = len(data)
    gscpi = data[:, -1]
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        if gscpi[t-1] > THRESHOLD:
            regime[t] = 1
    return data, regime, df.index

def run_lp_irfs(data_df, results, regime, h_max=24, target_var="ln_pce_goods"):
    irf_u = results.get("undisrupted")
    irf_d = results.get("disrupted")
    if irf_u is None or irf_d is None:
        return None, None
    idx = VAR_NAMES.index(target_var)
    lp_u = np.median(irf_u, axis=0)[:h_max+1, idx]
    lp_d = np.median(irf_d, axis=0)[:h_max+1, idx]
    return lp_u, lp_d

def plot_lp_irfs(lp_u, lp_d, h_max=24, save_path="figures/figure4_lp_irfs"):
    os.makedirs("figures", exist_ok=True)
    horizons = np.arange(h_max+1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(horizons, lp_u, "b-",  label="Undisrupted", linewidth=2)
    ax.plot(horizons, lp_d, "r--", label="Disrupted",   linewidth=2)
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Months after shock")
    ax.set_ylabel("Response (pp)")
    ax.set_title("LP Robustness: PCE Goods Response to MP Shock")
    ax.legend()
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{save_path}.{ext}", dpi=150)
    plt.close()
    print(f"Figure 4 saved: {save_path}.*")
