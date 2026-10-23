"""Comparison of upper-limit methods (profile likelihood vs nested sampling).

Main script accompanying the gammapy v2 paper (refactoring of
``UL-simulation-paper-script.ipynb``). The simulation itself lives in
``ul_simulation.py``; this script holds the configuration, runs (or reloads)
the simulation and makes the figures / summary numbers.

Run with::

    python ul_comparison_main.py
"""

import os

os.environ.setdefault("GAMMAPY_DATA", "/Users/facero/Documents/Work/Program/gammapy-data/dev")

import time

import numpy as np
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from matplotlib.ticker import MaxNLocator
from scipy.stats import median_abs_deviation as mad

from ul_simulation import default_cases, load_results, run_simulation, save_results, select

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

E1, E2 = 0.05 * u.TeV, 50 * u.TeV

# selection_optional is derived from the cases when absent (add "selection_optional": ["all"] to force it).
# The profile UL threshold is delta_ts = n_sigma_ul**2 (2.706 for 1.645).
fpe_config = {"n_sigma_ul": 1.645, "n_sigma_sensitivity": 2}

exposures = ["0.03 h", "0.3 h", "100 h"]
n_runs = 100
seed = "range"  # seeds 0..n_runs-1, the same every time you run the script. "random" for random seeds.
n_jobs = None  # None: all CPUs

# profile-zero, profile-negative, sampler-zero, sampler-negative.
# profile_ul_methods: any of "scipy" (default), "fpe", "interp".
cases = default_cases(profile_ul_methods=("scipy",), confidence_level_samples=0.954)
dataset_types = ("on", "asimov")

profile_ul_method = "scipy"  # profile UL method used in the plots
priors = ("zero", "negative")
prior_titles = {"zero": "Constrained (amplitude >= 0)", "negative": "Unconstrained (negative amplitude allowed)"}

outfile = f"uls_results_Nrun_{n_runs}.ecsv"
reload_from_disk = False  # True: skip the simulation and read `outfile`
show_plots = False


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"font.family": "serif"})

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# ---------------------------------------------------------------------------
# Plots and summary
# ---------------------------------------------------------------------------

def plot_lc_triangle(uls_profile, uls_sampler, histbins=25):
    """ULs per realization (left) and their distributions (right)."""
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[3, 1], wspace=0.05), sharey=True
    )
    alpha = 0.6

    uls_profile = np.asarray(uls_profile, dtype=float)
    uls_sampler = np.asarray(uls_sampler, dtype=float)

    with quantity_support():
        ax.scatter(np.arange(len(uls_profile)), uls_profile, alpha=alpha, marker="v", color="tab:orange", label="UL profile")
        ax.scatter(np.arange(len(uls_sampler)), uls_sampler, alpha=alpha, marker="o", color="tab:blue", label="UL sampler")

        qmed = 0.5
        ax.axhline(y=np.nanquantile(uls_profile, qmed), alpha=alpha, label=r"median", ls="--", color="tab:orange")
        ax.axhline(y=np.nanquantile(uls_sampler, qmed), alpha=alpha, label=r"median", ls="--", color="tab:blue")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=11))
    ax.set_xlabel("Realization")
    ax.set_ylabel(r"Differential Flux 95% UL [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
    ax.set_ylim(1e-16, 4e-12)
    ax.set_yscale("log")
    ax.legend()

    # side histogram of the ULs. Warning: there could be some negative values in the profile ULs
    values = np.concatenate([uls_profile, uls_sampler])
    values = values[np.isfinite(values) & (values > 0)]
    if len(values):
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), histbins)
        axh.hist(uls_sampler[np.isfinite(uls_sampler)], bins=bins, orientation="horizontal", color="tab:blue", alpha=alpha)
        axh.hist(uls_profile[np.isfinite(uls_profile)], bins=bins, orientation="horizontal", color="tab:orange", alpha=alpha)
    axh.set_xlabel("Occurence")
    axh.tick_params(labelleft=False)

    return ax


def get_uls(table, exposure, prior, method, dataset_type="on"):
    """UL values (cm-2 s-1 TeV-1) sorted by seed, for one exposure / prior / method."""
    ul_method = profile_ul_method if method == "profile" else "samples"
    rows = select(table, exposure=exposure, prior=prior, method=method, ul_method=ul_method, dataset_type=dataset_type)
    return rows["ul"].to_value("cm-2 s-1 TeV-1")


def plot_and_summarize(table):
    exposures = table.meta.get("exposures", list(np.unique(table["exposure"])))
    bkg_counts = table.meta["bkg_counts"]

    for prior in priors:
        print(f"\n===== prior: {prior} - {prior_titles.get(prior, prior)} =====")
        for exposure in exposures:
            uls_profile = get_uls(table, exposure, prior, "profile")
            uls_sampler = get_uls(table, exposure, prior, "sampler")
            asimov_profile = get_uls(table, exposure, prior, "profile", "asimov")
            asimov_sampler = get_uls(table, exposure, prior, "sampler", "asimov")

            print(f"{exposure}  background counts in ON region = {bkg_counts[exposure]:.2f}")
            failed = np.isnan(uls_profile).sum()
            print(f"Nb of Nan in LL profile:{failed} ({failed / len(uls_profile) * 100:.1f} %) ")

            ax = plot_lc_triangle(uls_profile, uls_sampler, histbins=20)

            if len(asimov_profile) and len(asimov_sampler):
                print(f"Bayes/profile Asimov = {asimov_sampler[0] / asimov_profile[0]:.2f}")
            m = np.isnan(uls_profile)
            print(f"MAD(UL_prof)/median(UL_prof) = {(mad(uls_profile[~m]) / np.median(uls_profile[~m])):.2f}")
            print(f"MAD(UL_sampler)/median(UL_sampler) = {(mad(uls_sampler) / np.median(uls_sampler)):.2f}")

            ax.set_title(f"exposure={exposure} ON-region bkg counts={bkg_counts[exposure]:.1f}")
            plt.savefig(f"UL_comparison_exp_{exposure}_prior_{prior}.pdf")
            if show_plots:
                plt.show()
            plt.close("all")

    # constrained / unconstrained ratios
    if {"zero", "negative"} <= set(priors):
        print("\n===== constrained / unconstrained =====")
        for exposure in exposures:
            print(exposure)
            prof = {p: get_uls(table, exposure, p, "profile") for p in ("zero", "negative")}
            samp = {p: get_uls(table, exposure, p, "sampler") for p in ("zero", "negative")}
            prof_asimov = {p: get_uls(table, exposure, p, "profile", "asimov") for p in ("zero", "negative")}
            samp_asimov = {p: get_uls(table, exposure, p, "sampler", "asimov") for p in ("zero", "negative")}
            if all(len(v) for v in (*prof_asimov.values(), *samp_asimov.values())):
                print(f"Profile constraint/profile unconstraint Asimov = {prof_asimov['zero'][0] / prof_asimov['negative'][0]:.2f}")
                print(f"Bayes constraint/Bayes unconstraint Asimov = {samp_asimov['zero'][0] / samp_asimov['negative'][0]:.2f}")
            print("--")
            print(f"median(UL_prof constr)/median(UL_prof unconstr) = {np.nanmedian(prof['zero']) / np.nanmedian(prof['negative']):.2f}")
            print(f"median(UL_sampler constr)/median(UL_sampler unconstr) = {np.nanmedian(samp['zero']) / np.nanmedian(samp['negative']):.2f}")
            print("")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not show_plots:
        matplotlib.use("Agg")
    set_style()

    if reload_from_disk:
        table = load_results(outfile)
        print(f"Loaded results from {outfile}")
    else:
        t0 = time.time()
        table = run_simulation(
            exposures, cases, n_runs, fpe_config, E1, E2, dataset_types=dataset_types, seed=seed, n_jobs=n_jobs
        )
        print(f"Simulation done in {time.time() - t0:.0f} s")
        save_results(table, outfile)
        print(f"Written results to {outfile}")

    n_failed = (~table["success"]).sum()
    if n_failed:
        print(f"Warning: {n_failed}/{len(table)} rows failed, e.g. {table['message'][~table['success']][0]}")

    plot_and_summarize(table)
