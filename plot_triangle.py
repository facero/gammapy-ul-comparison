"""ULs per realization with side histograms, as ``plot_lc_triangle`` of the original notebook.

One panel per exposure (rows: 0.03 h and 100 h) and constraint (columns: amplitude >= 0, unconstrained),
profile likelihood vs Bayesian ULs on the same realizations. Reads the main run of
``ul_report.py`` (``report_results/uls_main.ecsv``), so run that first.

Run with::

    python plot_triangle.py                # all realizations
    python plot_triangle.py --n-show 100   # scatter of the first 100 realizations only (histograms use all)

Outputs: ``report_figures/fig8_ul_per_realization.png`` (grid) and one PDF per
exposure and constraint (4 in total) in ``report_figures/triangle/``, named as in the notebook
(``UL_comparison_exp_<exposure>_prior_<zero|negative>.pdf``).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import median_abs_deviation as mad

from ul_comparison_main import set_style
from ul_report import figures_dir, get, results_dir
from ul_simulation import load_results

exposures = ["0.03 h", "100 h"]  # Poisson and Gaussian regimes (0.3 h skipped)
PRIORS = {"zero": "amplitude $\\geq 0$", "negative": "unconstrained"}
COLORS = {"profile": "tab:orange", "sampler": "tab:blue"}
MARKERS = {"profile": "v", "sampler": "o"}
LABELS = {"profile": "UL profile", "sampler": "UL Bayes"}


def plot_lc_triangle(ax, axh, uls, asimov, n_show=None, histbins=25, ylim=None):
    """Scatter of the ULs vs realization (``ax``) and their histograms (``axh``), log scale.

    ``uls`` and ``asimov`` are {"profile": array, "sampler": array / float}. Non-positive ULs
    cannot be shown on a log axis: their number is written in the panel.
    """
    alpha = 0.6
    n = len(uls["profile"])
    n_show = n if n_show is None else min(n_show, n)
    x = np.arange(n_show)
    ms = 12 if n_show <= 200 else 4
    for method in ("profile", "sampler"):
        v = uls[method]
        color = COLORS[method]
        ax.scatter(x, v[:n_show], s=ms, alpha=alpha, marker=MARKERS[method], color=color, label=LABELS[method], lw=0)
        ax.axhline(np.nanmedian(v), ls="--", color=color, label="median")
        ax.axhline(asimov[method], ls=":", color=color, lw=1.5, label="Asimov")

    values = np.concatenate([uls["profile"], uls["sampler"]])
    values = values[np.isfinite(values) & (values > 0)]
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), histbins)
    for method in ("sampler", "profile"):
        v = uls[method]
        axh.hist(v[np.isfinite(v) & (v > 0)], bins=bins, orientation="horizontal", color=COLORS[method], alpha=alpha)
        axh.axhline(np.nanmedian(v), ls="--", color=COLORS[method])

    text = []
    for method in ("profile", "sampler"):
        v = uls[method][np.isfinite(uls[method])]
        n_neg = int(np.sum(v <= 0))
        line = f"{LABELS[method]}: MAD/median = {mad(v) / np.median(v):.2f}"
        if n_neg:
            line += f", {n_neg} UL $\\leq 0$ not shown"
        text.append(line)
    text.append(f"Bayes/profile: median {np.nanmedian(uls['sampler']) / np.nanmedian(uls['profile']):.2f}, "
                f"Asimov {asimov['sampler'] / asimov['profile']:.2f}")
    ax.text(0.01, 0.02, "\n".join(text), transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(facecolor="w", alpha=0.8, lw=0))

    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlim(-0.01 * n_show, 1.01 * n_show)
    axh.tick_params(labelleft=False)


def get_case(table, exposure, prior):
    uls = {m: get(table, exposure, f"{m}-{prior}") for m in ("profile", "sampler")}
    asimov = {m: get(table, exposure, f"{m}-{prior}", dataset_type="asimov")[0] for m in ("profile", "sampler")}
    return uls, asimov


def row_ylim(table, exposure):
    """Common y range of a row (both constraints), from the positive ULs."""
    v = np.concatenate([get(table, exposure, f"{m}-{p}") for m in ("profile", "sampler") for p in PRIORS])
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.quantile(v, [0.001, 1.0])
    return lo / 2, hi * 1.5


def bkg_title(table, exposure, prior):
    return f"{exposure}, ON-region bkg counts = {table.meta['bkg_counts'][exposure]:.1f}, {PRIORS[prior]}"


def main(n_show=None):
    set_style()
    table = load_results(results_dir / "uls_main.ecsv")
    figures_dir.mkdir(exist_ok=True)
    (figures_dir / "triangle").mkdir(exist_ok=True)

    # grid: rows = exposures, columns = constraints, each made of a scatter + histogram pair
    fig = plt.figure(figsize=(18, 4.2 * len(exposures)))
    outer = fig.add_gridspec(len(exposures), len(PRIORS), wspace=0.12, hspace=0.35)
    for i, exposure in enumerate(exposures):
        ylim = row_ylim(table, exposure)
        for j, prior in enumerate(PRIORS):
            inner = outer[i, j].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
            ax = fig.add_subplot(inner[0])
            axh = fig.add_subplot(inner[1], sharey=ax)
            uls, asimov = get_case(table, exposure, prior)
            plot_lc_triangle(ax, axh, uls, asimov, n_show=n_show, ylim=ylim)
            ax.set_title(bkg_title(table, exposure, prior), fontsize=11)
            if j == 0:
                ax.set_ylabel("95% UL [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
            if i == len(exposures) - 1:
                ax.set_xlabel("Realization")
                axh.set_xlabel("Occurrence")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, ncol=3, loc="upper left")
    path = figures_dir / "fig8_ul_per_realization.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Written {path}")

    # one figure per exposure and constraint, as in the notebook
    for exposure in exposures:
        ylim = row_ylim(table, exposure)
        for prior in PRIORS:
            fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 5), sharey=True,
                                          gridspec_kw=dict(width_ratios=[3, 1], wspace=0.05))
            uls, asimov = get_case(table, exposure, prior)
            plot_lc_triangle(ax, axh, uls, asimov, n_show=n_show, ylim=ylim)
            ax.set_title(bkg_title(table, exposure, prior))
            ax.set_xlabel("Realization")
            ax.set_ylabel(r"Differential Flux 95% UL [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
            axh.set_xlabel("Occurrence")
            ax.legend(fontsize=9, ncol=3, loc="upper left")
            path = figures_dir / "triangle" / f"UL_comparison_exp_{exposure}_prior_{prior}.pdf"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
    print(f"Written {len(exposures) * len(PRIORS)} PDFs in {figures_dir / 'triangle'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-show", type=int, default=None, help="number of realizations in the scatter plots")
    args = parser.parse_args()
    main(args.n_show)
