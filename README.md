# Replication: Zanetti et al. (2026)

Replication of **"Monetary Policy and Supply Chain Disruptions"** (Zanetti, Oxford, 2026), extended to India.

## What this does

- Estimates a two-regime Threshold VAR (TVAR) on US data (Jan 2017 – Sep 2023)
- Identifies monetary policy shocks via sign restrictions (Uhlig 2005)
- Compares IRFs across disrupted vs undisrupted supply chain regimes
- Robustness check via Local Projections (Jordà 2005)
- Extension: applies framework to India using RBI policy rate and CPI goods

## Regime definition

Disrupted regime: GSCPI_{t-1} > 3.34 (NY Fed Global Supply Chain Pressure Index)
Sample disrupted period: Jan 2022 – Sep 2022 (9 months)

## Key result

Monetary policy significantly less effective at compressing goods prices during supply chain disruptions. PCE goods response: -0.18pp (undisrupted) vs near-zero (disrupted).

## Data sources

- FRED (US macro): via FRED API
- GSCPI: Federal Reserve Bank of New York (Benigno et al. 2022), stored in `data/raw/gscpi.csv`
- India: FRED (INDCPIALLMINMEI, INDPROINDMISMEI, INTDSRINM193N)

## Structure

zanetti/
tvar_estimation.py # Bayesian TVAR, Gibbs sampler, sign restrictions
local_projections.py # Jordà LP robustness
data/
raw/gscpi.csv # NY Fed GSCPI series
processed/ # built by Cell 2-3 of notebook
figures/ # output figures
results/ # saved posterior draws

## Requirements

pip install pandas numpy scipy matplotlib requests


Set `FRED_API_KEY` as an environment variable before running.

## Reference

Zanetti, F. et al. (2026). *Monetary Policy and Supply Chain Disruptions*. University of Oxford.
Benigno, G. et al. (2022). Global Supply Chain Pressure Index. FRBNY Staff Reports.
