"""Spread of the upper limits at zero signal: Brazil bands, simulations vs Gaussian expectation.

For background-only realizations (mu = 0), compares the observed quantiles of
UL / sigma_A (median, +-1 sigma, +-2 sigma, i.e. quantiles 2.3, 15.9, 50, 84.1
and 97.7 %) with the expected-limit bands of the Gaussian (Wilks) limit
(Cowan, Cranmer, Gross & Vitells 2011).

Expected bands: under mu = 0 the best fit is z = mu_hat / sigma_A ~ N(0, 1) and
the UL is a monotonic function UL(z) (``theory_ul`` in ``ul_report.py``, with
the lower bound of each case). So the quantile of the UL at Phi(N) is simply
UL(z = N), for N = -2, -1, 0, 1, 2.

Two exposures: 0.03 h (~10 background counts, Poisson regime) and 100 h
(~33 500 counts, Gaussian regime). The Bayesian ULs use the exact "grid"
posterior quantile (no nested sampling noise, see Fig. 6 of the report).

Run with::

    python brazil_bands.py

Outputs: ``report_results/uls_brazil.ecsv``, ``report_results/brazil_summary.json``,
``report_figures/fig7_brazil_bands.png``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as gauss

from ul_comparison_main import set_style
from ul_report import (
    CASE_BOUNDS, CL_SAMPLES, E1, E2, STYLE, UL_METHOD, fpe_config, figures_dir, get, results_dir, sigma_asimov, theory_ul,
)
from ul_simulation import (
    asimov_dataset, build_simulation, default_cases, load_results, run_case, run_simulation, save_results,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

exposures = ["0.03 h", "100 h"]
n_runs = 1000  # +-2 sigma quantiles (2.3 %) need ~1000 realizations
seed = 2000  # seeds 2000..2999, independent of the report runs
recompute = False  # False: reload report_results/uls_brazil.ecsv when it exists
n_boot = 2000  # bootstrap resamples for the uncertainty of the observed quantiles

cases = default_cases(profile_ul_methods=("scipy",), sampler_ul_methods=("grid",),
                      confidence_level_samples=CL_SAMPLES)
case_names = [c.name for c in cases]

N_SIGMA = np.array([-2, -1, 0, 1, 2])
QUANTILES = gauss.cdf(N_SIGMA)  # 0.023, 0.159, 0.5, 0.841, 0.977

results_file = results_dir / "uls_brazil.ecsv"
summary_file = results_dir / "brazil_summary.json"
figure_file = figures_dir / "fig7_brazil_bands.png"


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

def run_or_load():
    if results_file.exists() and not recompute:
        print(f"Loading {results_file}")
        return load_results(results_file)
    results_file.parent.mkdir(exist_ok=True)
    t0 = time.time()
    table = run_simulation(exposures, cases, n_runs, fpe_config, E1, E2, dataset_types=("on", "asimov"), seed=seed)
    print(f"Brazil run: {time.time() - t0:.0f} s")
    save_results(table, results_file)
    return table


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def expected_bands(case, sigma):
    """UL / sigma_A at the quantiles Phi(N) in the Gaussian limit."""
    return theory_ul(case, N_SIGMA.astype(float), CASE_BOUNDS[case] / sigma)


def observed_bands(x, rng):
    """Observed quantiles of x and their bootstrap standard deviation."""
    x = x[np.isfinite(x)]
    q = np.quantile(x, QUANTILES)
    boot = np.quantile(rng.choice(x, size=(n_boot, x.size)), QUANTILES, axis=1)
    return q, boot.std(axis=1)


def band_fractions(x, expected):
    """Fraction of ULs inside the expected +-1 and +-2 sigma bands (68.3 % and 95.4 % expected)."""
    x = x[np.isfinite(x)]
    return {
        "in_1sigma": float(np.mean((x >= expected[1]) & (x <= expected[3]))),
        "in_2sigma": float(np.mean((x >= expected[0]) & (x <= expected[4]))),
        "below_2sigma": float(np.mean(x < expected[0])),
        "above_2sigma": float(np.mean(x > expected[4])),
    }


def shifted_asimov_bands(sig):
    """UL / sigma_A on Asimov datasets with an injected amplitude N sigma_A (N = -2..2).

    Uses the true (Poisson) likelihood shape around a best fit at N sigma_A instead of
    the parabolic approximation, but still assumes that the best fit is Gaussian,
    mu_hat ~ N(0, sigma_A). Negative expected counts (only possible for N < 0 at very
    low counts) are clipped at 0 and counted in ``clipped``.
    """
    out = {}
    for exposure in exposures:
        s = sig[exposure]
        out[exposure] = {case: [] for case in case_names}
        out[exposure]["clipped"] = []
        for n in N_SIGMA:
            dataset, fit_model, simu_model = build_simulation(exposure, E1, E2, simu_amplitude=n * s if n else 1e-20)
            asimov = asimov_dataset(dataset, simu_model)
            mask = asimov.counts.data < 0
            out[exposure]["clipped"].append(int(mask[dataset.mask.data].sum()))
            asimov.counts.data[mask] = 0
            for case in cases:
                uls, _ = run_case(case, asimov, fit_model, fpe_config, E1, E2)
                out[exposure][case.name].append(float(uls[UL_METHOD[case.name]][0]) / s)
        print(f"shifted Asimov {exposure}: done")
    return out


def compute(table, sig, shifted):
    rng = np.random.default_rng(0)
    out = {}
    for exposure in exposures:
        s = sig[exposure]
        d = {"bkg_counts": float(table.meta["bkg_counts"][exposure]), "sigma_A": s}
        for case in case_names:
            x = get(table, exposure, case) / s
            asimov = get(table, exposure, case, dataset_type="asimov") / s
            q, dq = observed_bands(x, rng)
            exp = expected_bands(case, s)
            d[case] = {
                "n": int(np.isfinite(x).sum()),
                "n_nan": int((~np.isfinite(x)).sum()),
                "observed": q.tolist(),
                "observed_err": dq.tolist(),
                "expected": exp.tolist(),
                "asimov": float(asimov[0]),
                "shifted_asimov": shifted[exposure][case],
                "observed_over_expected": (q / exp).tolist(),
                **band_fractions(x, exp),
            }
        out[exposure] = d
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

SHORT_LABEL = {
    "profile-zero": "profile\namplitude $\\geq 0$",
    "profile-negative": "profile\nunconstrained",
    "sampler-zero": "Bayes\nprior $\\geq 0$",
    "sampler-negative": "Bayes\nprior $\\geq -3\\times10^{-14}$",
}


def fig_brazil(bands, table):
    """One panel per exposure, one column per case: expected bands (green/yellow, left) vs observed (right)."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(exposures), figsize=(6.5 * len(exposures), 5.4), sharey=True)
    w = 0.34
    ytop = 0.9 + max(max(np.max(b["expected"]), np.max(np.add(b["observed"], b["observed_err"])),
                         np.max(b["shifted_asimov"])) for e in exposures for c, b in bands[e].items() if c in case_names)
    for ax, exposure in zip(np.atleast_1d(axes), exposures):
        for i, case in enumerate(case_names):
            b = bands[exposure][case]
            e, o, do = np.array(b["expected"]), np.array(b["observed"]), np.array(b["observed_err"])
            color = STYLE[case]["color"]
            # expected bands, left box
            xe = [i - w - 0.02, i - 0.02]
            ax.fill_between(xe, e[0], e[4], color="#ffd400", lw=0)
            ax.fill_between(xe, e[1], e[3], color="#2ca02c", lw=0)
            ax.plot(xe, [e[2]] * 2, "k--", lw=1.2)
            ax.plot([i - w / 2 - 0.02] * 5, b["shifted_asimov"], marker="D", color="w", mec="k", ms=5, ls="none", zorder=5)
            # observed quantiles, right box (same layout, case colour)
            xo = [i + 0.02, i + w + 0.02]
            ax.fill_between(xo, o[0], o[4], color=color, alpha=0.3, lw=0)
            ax.fill_between(xo, o[1], o[3], color=color, alpha=0.75, lw=0)
            ax.plot(xo, [o[2]] * 2, "k-", lw=1.5)
            # bootstrap uncertainty of each observed quantile
            ax.errorbar([i + w / 2 + 0.02] * 5, o, yerr=do, fmt="none", ecolor="k", capsize=2.5, lw=0.8)
            ax.plot(i, b["asimov"], marker="*", color="w", mec="k", ms=11, ls="none", zorder=5)
            ax.text(i, ytop - 0.08, f"in $\\pm1\\sigma$: {b['in_1sigma']:.2f}\nin $\\pm2\\sigma$: {b['in_2sigma']:.2f}",
                    ha="center", va="top", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(case_names)))
        ax.set_xticklabels([SHORT_LABEL[c] for c in case_names], fontsize=9)
        ax.set_xlim(-0.55, len(case_names) - 0.45)
        ax.set_title(f"{exposure}, $N_\\mathrm{{bkg}}$={table.meta['bkg_counts'][exposure]:.0f}, "
                     f"{bands[exposure][case_names[0]]['n']} realizations")
        ax.grid(axis="y", alpha=0.3)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylim(-0.6, ytop)
    ax0.set_ylabel(r"UL / $\sigma_A$")
    handles = [
        Patch(color="#2ca02c", label="expected $\\pm1\\sigma$ (Gaussian limit)"),
        Patch(color="#ffd400", label="expected $\\pm2\\sigma$"),
        Line2D([], [], color="k", ls="--", label="expected median"),
        Line2D([], [], marker="D", color="w", mec="k", ms=5, ls="none",
               label="Asimov with injected $N\\sigma_A$, $N=-2..2$"),
        Patch(color="grey", alpha=0.75, label="observed 15.9-84.1 %"),
        Patch(color="grey", alpha=0.3, label="observed 2.3-97.7 %"),
        Line2D([], [], color="k", label="observed median ($\\pm$ bootstrap error)"),
        Line2D([], [], marker="*", color="w", mec="k", ms=11, ls="none", label="Asimov dataset"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("UL spread at zero signal: expected Brazil bands (left box) vs observed quantiles (right box)", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(figure_file, dpi=150)
    print(f"Written {figure_file}")
    return fig


def print_table(bands):
    labels = ["-2s", "-1s", "med", "+1s", "+2s"]
    for exposure in exposures:
        d = bands[exposure]
        print(f"\n=== {exposure}  (N_bkg = {d['bkg_counts']:.0f}, sigma_A = {d['sigma_A']:.3e})")
        for case in case_names:
            b = d[case]
            obs = "  ".join(f"{o:5.2f}±{e:4.2f}" for o, e in zip(b["observed"], b["observed_err"]))
            exp = "  ".join(f"{x:9.2f}" for x in b["expected"])
            print(f"{case:17s} observed  {obs}")
            print(f"{'':17s} expected  {exp}     ({', '.join(labels)})")
            sh = "  ".join(f"{x:9.2f}" for x in b["shifted_asimov"])
            print(f"{'':17s} shifted   {sh}")
            print(f"{'':17s} in ±1σ {b['in_1sigma']:.3f} (0.683)  in ±2σ {b['in_2sigma']:.3f} (0.954)  "
                  f"below/above ±2σ {b['below_2sigma']:.3f}/{b['above_2sigma']:.3f} (0.023)  NaN {b['n_nan']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-runs", type=int, default=n_runs)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="write results and figure here instead of report_results/ and report_figures/ (tests)")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    n_runs, recompute = args.n_runs, args.recompute or recompute
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        results_file = args.out_dir / "uls_brazil.ecsv"
        summary_file = args.out_dir / "brazil_summary.json"
        figure_file = args.out_dir / "fig7_brazil_bands.png"
    set_style()
    figures_dir.mkdir(exist_ok=True)
    table = run_or_load()
    sig = sigma_asimov(table)
    import warnings
    import logging
    warnings.simplefilter("ignore")
    logging.getLogger("gammapy").setLevel(logging.ERROR)
    shifted = shifted_asimov_bands(sig)
    bands = compute(table, sig, shifted)
    bands["shifted_asimov_clipped_bins"] = {e: shifted[e]["clipped"] for e in exposures}
    fig_brazil(bands, table)
    print_table(bands)
    with open(summary_file, "w") as f:
        json.dump(bands, f, indent=1)
