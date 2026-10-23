# Upper limits in gammapy: profile likelihood vs Bayesian (nested sampling)

Simulation study comparing upper-limit (UL) estimators for a faint source with
gammapy (v2 dev) and CTAO-South Prod5 IRFs. The study covers the profile
likelihood (`FluxPointsEstimator` / scipy) and the Bayesian posterior quantile
(UltraNest nested sampling or exact grid integration), each with physical
(≥ 0) or "unconstrained" lower bounds.

The full write-up is [`UL_report.md`](UL_report.md) ([PDF](UL_report.pdf)).

## Contents

| File | Purpose |
|---|---|
| `ul_simulation.py` | Library: datasets, test cases (`Case`, `default_cases`), parallel simulation driver (`run_simulation`), I/O (`save_results`, `load_results`, `select`) |
| `ul_comparison_main.py` | Refactored version of the original notebook: paper figures |
| `ul_report.py` | Simulations and figures 1–6 of the report (spread, Bayes/profile ratio, coverage, sampler noise) |
| `brazil_bands.py` | Section 5.6 of the report: Brazil bands at zero signal (Gaussian Asimov, shifted Asimov, toys of the main run) |
| `UL-simulation-paper-script.ipynb` | Original notebook |
| `report_results/` | Raw results (ECSV, one row per realization × case × UL method) and JSON summaries |
| `report_figures/` | Figures of the report |

## Running

Requires gammapy (dev), UltraNest, scipy, matplotlib, tqdm and `GAMMAPY_DATA`
(the scripts set a default path, change it for your machine).

```
python ul_report.py      # reloads report_results/*.ecsv if present, otherwise reruns them (~45 min on 8 cores)
python brazil_bands.py   # Brazil bands from report_results/uls_main.ecsv (~1 min)
```

Realizations are distributed over `N_JOBS` processes (`ul_report.py`, `n_jobs` in `run_simulation`).
