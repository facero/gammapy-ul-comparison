"""Simulations and figures for the profile likelihood vs Bayesian (sampler) UL report.

Runs (or reloads from ``report_results/``):

1. main: background-only realizations for 3 exposures, the 4 default cases,
   Bayesian ULs from both nested sampling ("samples") and exact grid integration ("grid"),
2. coverage: same with an injected signal of 1 and 3 sigma_A,
3. sampler noise: the same faked dataset analysed N times by the sampler.

and makes the figures of ``UL_report.md`` in ``report_figures/``.

sigma_A ("Asimov sigma") is the 1-sigma amplitude uncertainty from the Asimov
background-only dataset: sigma_A = UL_Asimov(profile) / sqrt(delta_ts).

Run with::

    python ul_report.py
"""

import os

os.environ.setdefault("GAMMAPY_DATA", "/Users/facero/Documents/Work/Program/gammapy-data/dev")

import json
import time
from pathlib import Path

import numpy as np
import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import vstack
from scipy.stats import median_abs_deviation as mad
from scipy.stats import norm as gauss

from ul_comparison_main import set_style
from ul_simulation import default_cases, load_results, run_simulation, save_results, select

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

E1, E2 = 0.05 * u.TeV, 50 * u.TeV
fpe_config = {"n_sigma_ul": 1.645, "n_sigma_sensitivity": 2}
DELTA_TS = fpe_config["n_sigma_ul"] ** 2
CL_SAMPLES = 0.954
REF_AMPLITUDE = 1e-13  # amplitude of the fit model = reference of the FluxPointsEstimator norm

exposures = ["0.03 h", "0.3 h", "100 h"]
n_runs = 100
coverage_exposures = ["0.3 h", "100 h"]
coverage_levels = [1.0, 3.0]  # injected amplitude in units of sigma_A
n_repeat = 30  # sampler noise test: number of sampler runs on the same dataset

cases = default_cases(profile_ul_methods=("scipy",), sampler_ul_methods=("samples", "grid"),
                      confidence_level_samples=CL_SAMPLES)
CASE_BOUNDS = {c.name: (c.norm_min * REF_AMPLITUDE if c.method == "profile" else c.amplitude_min) for c in cases}

results_dir = Path("report_results")
figures_dir = Path("report_figures")
recompute = False  # False: reload the ECSV files when they exist

STYLE = {
    "profile-zero": dict(color="tab:orange", marker="v", label="profile, amplitude $\\geq 0$"),
    "profile-negative": dict(color="tab:red", marker="^", label="profile, unconstrained"),
    "sampler-zero": dict(color="tab:blue", marker="o", label="Bayes, uniform prior $\\geq 0$"),
    "sampler-negative": dict(color="tab:cyan", marker="s", label="Bayes, uniform prior $\\geq -3\\times10^{-14}$"),
}
UL_METHOD = {"profile-zero": "scipy", "profile-negative": "scipy", "sampler-zero": "grid", "sampler-negative": "grid"}


# ---------------------------------------------------------------------------
# Gaussian (Wilks) expectations, in units of sigma: z = best fit / sigma, l / a = lower bound / sigma
# ---------------------------------------------------------------------------

def theory_profile_ul(z, l=-np.inf, delta_ts=DELTA_TS):
    """Profile UL for a parabolic -2 log L, best fit z, parameter bounded at l (z may be < l)."""
    z = np.asarray(z, dtype=float)
    k2 = delta_ts
    return np.where(z >= l, z + np.sqrt(k2), z + np.sqrt((l - z) ** 2 + k2))


def theory_bayes_ul(z, a=-np.inf, cl=CL_SAMPLES):
    """Quantile ``cl`` of a Gaussian likelihood N(z, 1) times a uniform prior on [a, +inf)."""
    z = np.asarray(z, dtype=float)
    pa = gauss.cdf(a - z)
    return z + gauss.ppf(pa + cl * (1 - pa))


def theory_ul(case, z, bound):
    return theory_profile_ul(z, bound) if case.startswith("profile") else theory_bayes_ul(z, bound)


def theory_coverage(case, mu, bound, n=200000, rng=np.random.default_rng(1)):
    """P(UL >= mu) for z ~ N(mu, 1)."""
    z = rng.normal(mu, 1, n)
    return np.mean(theory_ul(case, z, bound) >= mu)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

def run_or_load(filename, func):
    path = results_dir / filename
    if path.exists() and not recompute:
        print(f"Loading {path}")
        return load_results(path)
    t0 = time.time()
    table = func()
    print(f"{filename}: {time.time() - t0:.0f} s")
    save_results(table, path)
    return table


def sigma_asimov(table):
    """sigma_A per exposure from the Asimov constrained profile UL."""
    out = {}
    for exposure in np.unique(table["exposure"]):
        ul = select(table, exposure=exposure, dataset_type="asimov", case="profile-zero", ul_method="scipy")["ul"]
        out[str(exposure)] = float(ul.to_value("cm-2 s-1 TeV-1")[0]) / np.sqrt(DELTA_TS)
    return out


def run_all():
    results_dir.mkdir(exist_ok=True)
    main = run_or_load(
        "uls_main.ecsv",
        lambda: run_simulation(exposures, cases, n_runs, fpe_config, E1, E2, dataset_types=("on", "asimov")),
    )
    sig = sigma_asimov(main)

    def coverage_run():
        tables = []
        for level in coverage_levels:
            amp = {e: level * sig[e] for e in coverage_exposures}
            t = run_simulation(coverage_exposures, cases, n_runs, fpe_config, E1, E2, dataset_types=("on",),
                               simu_amplitude=amp, seed=1000)  # seeds different from the main run
            t["level"] = level
            tables.append(t)
        table = vstack(tables, metadata_conflicts="silent")
        return table

    coverage = run_or_load("uls_coverage.ecsv", coverage_run)

    sampler_cases = [c for c in cases if c.method == "sampler"]
    noise = run_or_load(
        "uls_sampler_noise.ecsv",
        lambda: run_simulation(coverage_exposures, sampler_cases, n_repeat, fpe_config, E1, E2,
                               dataset_types=("on",), seed=[0] * n_repeat),
    )
    return main, coverage, noise, sig


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def get(table, exposure, case, ul_method=None, dataset_type="on", column="ul", **filters):
    ul_method = ul_method or UL_METHOD[case]
    rows = select(table, exposure=exposure, case=case, ul_method=ul_method, dataset_type=dataset_type, **filters)
    values = rows[column]
    return values.to_value("cm-2 s-1 TeV-1") if isinstance(values, u.Quantity) else np.asarray(values)


def bkg_label(table, exposure):
    return f"{exposure}, $N_\\mathrm{{bkg}}$={table.meta['bkg_counts'][exposure]:.0f}"


def fig_ul_vs_bestfit(main, sig, summary):
    """UL / sigma_A vs maximum likelihood amplitude / sigma_A, with Gaussian expectations."""
    fig, axes = plt.subplots(1, len(exposures), figsize=(5 * len(exposures), 4.5), sharey=True)
    for ax, exposure in zip(axes, exposures):
        s = sig[exposure]
        # unconstrained-ish best fit, shared by all cases of the same seed
        z = get(main, exposure, "profile-negative", column="best_fit") / s
        zz = np.linspace(min(-3, np.nanmin(z)), max(3, np.nanmax(z)), 300)
        for case in STYLE:
            ul = get(main, exposure, case) / s
            st = STYLE[case]
            ax.scatter(z, ul, s=12, alpha=0.6, color=st["color"], marker=st["marker"], label=st["label"])
            ax.plot(zz, theory_ul(case, zz, CASE_BOUNDS[case] / s), color=st["color"], lw=1.5)
        ax.axvline(CASE_BOUNDS["profile-negative"] / s, color="grey", ls=":", lw=1)
        ax.set_title(bkg_label(main, exposure))
        ax.set_xlabel(r"best-fit amplitude (profile, unconstr.) / $\sigma_A$")
        ax.set_xlim(zz[0], zz[-1])
    axes[0].set_ylabel(r"UL / $\sigma_A$")
    axes[0].set_ylim(0, 6)
    axes[0].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_ul_vs_bestfit.png", dpi=150)
    plt.close(fig)


def fig_paired(main, sig, summary):
    """Per realization: Bayesian UL vs profile UL (same faked data)."""
    fig, axes = plt.subplots(1, len(exposures), figsize=(5 * len(exposures), 4.8))
    zz = np.linspace(-4, 4, 400)
    for ax, exposure in zip(axes, exposures):
        s = sig[exposure]
        for prior, (cp, cs) in {"zero": ("profile-zero", "sampler-zero"),
                                "negative": ("profile-negative", "sampler-negative")}.items():
            x = get(main, exposure, cp) / s
            y = get(main, exposure, cs) / s
            st = STYLE[cs]
            ax.scatter(x, y, s=12, alpha=0.6, color=st["color"], marker=st["marker"], label=f"prior {prior}")
            ax.plot(theory_ul(cp, zz, CASE_BOUNDS[cp] / s), theory_ul(cs, zz, CASE_BOUNDS[cs] / s),
                    color=st["color"], lw=1.5)
        lim = [0, max(4.5, np.nanmax(get(main, exposure, "sampler-zero") / s) * 1.05)]
        ax.plot(lim, lim, color="k", lw=0.8, ls="--", label="1:1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(bkg_label(main, exposure))
        ax.set_xlabel(r"profile UL / $\sigma_A$")
        ax.set_ylabel(r"Bayesian UL / $\sigma_A$")
        ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_paired_bayes_vs_profile.png", dpi=150)
    plt.close(fig)


def fig_ecdf(main, sig, summary, n_mc=200000):
    """Cumulative distribution of UL / sigma_A: simulations vs Gaussian expectation."""
    rng = np.random.default_rng(2)
    zmc = rng.normal(0, 1, n_mc)
    fig, axes = plt.subplots(1, len(exposures), figsize=(5 * len(exposures), 4.2), sharey=True)
    for ax, exposure in zip(axes, exposures):
        s = sig[exposure]
        for case, st in STYLE.items():
            ul = np.sort(get(main, exposure, case) / s)
            ul = ul[np.isfinite(ul)]
            ax.step(ul, np.arange(1, len(ul) + 1) / len(ul), where="post", color=st["color"], label=st["label"])
            th = np.sort(theory_ul(case, zmc, CASE_BOUNDS[case] / s))
            ax.plot(th, np.arange(1, n_mc + 1) / n_mc, color=st["color"], ls="--", lw=1)
        ax.set_xlim(0, 5)
        ax.set_title(bkg_label(main, exposure))
        ax.set_xlabel(r"UL / $\sigma_A$")
    axes[0].set_ylabel("cumulative fraction of realizations")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_ecdf.png", dpi=150)
    plt.close(fig)


def fig_vs_bkg(main, sig, summary):
    """Bayes/profile ratios and spread vs background counts."""
    bkg = np.array([main.meta["bkg_counts"][e] for e in exposures])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for prior, (cp, cs) in {"zero": ("profile-zero", "sampler-zero"),
                            "negative": ("profile-negative", "sampler-negative")}.items():
        st = STYLE[cs]
        med = [np.nanmedian(get(main, e, cs) / get(main, e, cp)) for e in exposures]
        asi = [get(main, e, cs, dataset_type="asimov")[0] / get(main, e, cp, dataset_type="asimov")[0] for e in exposures]
        th = [theory_ul(cs, 0.0, CASE_BOUNDS[cs] / sig[e]) / theory_ul(cp, 0.0, CASE_BOUNDS[cp] / sig[e]) for e in exposures]
        ax1.plot(bkg, med, marker="o", color=st["color"], label=f"prior {prior}: median of per-realization ratio")
        ax1.plot(bkg, asi, marker="*", ms=10, ls=":", color=st["color"], label=f"prior {prior}: Asimov")
        ax1.plot(bkg, th, ls="--", color=st["color"], lw=1, label=f"prior {prior}: Gaussian expectation (Asimov)")
        for case, ls in [(cp, "-"), (cs, "--")]:
            spread = []
            for e in exposures:
                v = get(main, e, case)
                v = v[np.isfinite(v)]
                spread.append(mad(v) / np.median(v))
            ax2.plot(bkg, spread, marker=STYLE[case]["marker"], ls=ls, color=STYLE[case]["color"], label=STYLE[case]["label"])
    ax1.axhline(1, color="k", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("expected background counts in ON region")
    ax1.set_ylabel("Bayesian UL / profile UL")
    ax1.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("expected background counts in ON region")
    ax2.set_ylabel("MAD / median of the ULs")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_ratio_vs_background.png", dpi=150)
    plt.close(fig)


def coverage_fraction(table, exposure, case, mu, **filters):
    ul = get(table, exposure, case, **filters)
    ok = np.isfinite(ul)
    k, n = np.sum(ul[ok] >= mu), ok.sum()
    p = k / n
    return p, np.sqrt(max(p * (1 - p), 1 / n) / n), n


def fig_coverage(main, coverage, sig, summary):
    mus = np.linspace(0, 4, 41)
    fig, axes = plt.subplots(1, len(coverage_exposures), figsize=(6 * len(coverage_exposures), 4.5), sharey=True)
    summary["coverage"] = {}
    for ax, exposure in zip(axes, coverage_exposures):
        s = sig[exposure]
        summary["coverage"][exposure] = {}
        for i, (case, st) in enumerate(STYLE.items()):
            ax.plot(mus, [theory_coverage(case, m, CASE_BOUNDS[case] / s) for m in mus], color=st["color"], lw=1.2)
            pts = [(0.0, *coverage_fraction(main, exposure, case, 0.0))]
            for level in coverage_levels:
                mu_true = select(coverage, exposure=exposure, level=level)["simu_amplitude"][0].to_value("cm-2 s-1 TeV-1")
                pts.append((level, *coverage_fraction(coverage, exposure, case, mu_true, level=level)))
            x, p, e, n = map(np.array, zip(*pts))
            ax.errorbar(x + 0.04 * (i - 1.5), p, e, color=st["color"], marker=st["marker"], ls="none", label=st["label"])
            summary["coverage"][exposure][case] = {f"{xx:g}": [float(pp), float(ee)] for xx, pp, ee in zip(x, p, e)}
        ax.axhline(0.95, color="k", lw=0.8, ls="--")
        ax.set_title(bkg_label(main, exposure))
        ax.set_xlabel(r"injected amplitude / $\sigma_A$")
    axes[0].set_ylabel(r"coverage  P(UL $\geq$ true amplitude)")
    axes[0].set_ylim(0.8, 1.01)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_coverage.png", dpi=150)
    plt.close(fig)


def fig_sampler_noise(noise, summary):
    """Same dataset analysed n_repeat times: nested sampling UL scatter around the exact (grid) UL."""
    fig, axes = plt.subplots(1, len(coverage_exposures), figsize=(6 * len(coverage_exposures), 4), sharey=True)
    summary["sampler_noise"] = {}
    for ax, exposure in zip(axes, coverage_exposures):
        summary["sampler_noise"][exposure] = {}
        for case in ("sampler-zero", "sampler-negative"):
            samp = get(noise, exposure, case, ul_method="samples", sort=None)
            grid = get(noise, exposure, case, ul_method="grid", sort=None)
            rel = samp / grid[0] - 1
            st = STYLE[case]
            ax.hist(rel * 100, bins=np.linspace(-20, 20, 21), alpha=0.5, color=st["color"], label=st["label"])
            summary["sampler_noise"][exposure][case] = dict(
                rel_std=float(np.std(rel)), rel_mean=float(np.mean(rel)), grid_spread=float(np.ptp(grid) / grid[0])
            )
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{exposure}: {n_repeat} sampler runs on one dataset")
        ax.set_xlabel("(nested sampling UL - exact UL) / exact UL [%]")
    axes[0].set_ylabel("runs")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_sampler_noise.png", dpi=150)
    plt.close(fig)


def numbers(main, sig, summary):
    """Summary statistics for the report."""
    out = {}
    for exposure in exposures:
        s = sig[exposure]
        d = dict(bkg_counts=main.meta["bkg_counts"][exposure], sigma_A=s)
        for case in STYLE:
            ul = get(main, exposure, case)
            d[case] = dict(
                median_over_sigma=float(np.nanmedian(ul) / s),
                mad_over_median=float(mad(ul[np.isfinite(ul)]) / np.nanmedian(ul)),
                asimov_over_sigma=float(get(main, exposure, case, dataset_type="asimov")[0] / s),
                theory_asimov_over_sigma=float(theory_ul(case, 0.0, CASE_BOUNDS[case] / s)),
                theory_median_over_sigma=float(np.median(theory_ul(case, np.random.default_rng(3).normal(0, 1, 200000), CASE_BOUNDS[case] / s))),
                bound_over_sigma=float(CASE_BOUNDS[case] / s),
                n_nan=int(np.isnan(ul).sum()),
                n_scan_extended=int(np.char.startswith(np.asarray(select(main, exposure=exposure, case=case, ul_method=UL_METHOD[case], dataset_type="on")["message"], str), "norm scan extended").sum()),
            )
        for cs in ("sampler-zero", "sampler-negative"):
            samp = get(main, exposure, cs, ul_method="samples")
            grid = get(main, exposure, cs, ul_method="grid")
            d[cs]["samples_vs_grid_median_rel"] = float(np.nanmedian(samp / grid - 1))
            d[cs]["samples_vs_grid_std_rel"] = float(np.nanstd(samp / grid - 1))
        for prior, (cp, cs) in {"zero": ("profile-zero", "sampler-zero"), "negative": ("profile-negative", "sampler-negative")}.items():
            r = get(main, exposure, cs) / get(main, exposure, cp)
            d[f"ratio_{prior}"] = dict(median=float(np.nanmedian(r)), min=float(np.nanmin(r)), max=float(np.nanmax(r)),
                                       frac_bayes_larger=float(np.nanmean(r > 1)))
        d["constr_over_unconstr"] = {
            m: float(np.nanmedian(get(main, exposure, f"{m}-zero")) / np.nanmedian(get(main, exposure, f"{m}-negative")))
            for m in ("profile", "sampler")
        }
        out[exposure] = d
    summary["numbers"] = out


if __name__ == "__main__":
    set_style()
    figures_dir.mkdir(exist_ok=True)
    main, coverage, noise, sig = run_all()
    summary = {"sigma_A": sig}
    numbers(main, sig, summary)
    fig_ul_vs_bestfit(main, sig, summary)
    fig_paired(main, sig, summary)
    fig_ecdf(main, sig, summary)
    fig_vs_bkg(main, sig, summary)
    fig_coverage(main, coverage, sig, summary)
    fig_sampler_noise(noise, summary)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
