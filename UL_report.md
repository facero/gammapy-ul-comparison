# Upper limits on a faint gamma-ray source: profile likelihood vs Bayesian nested sampling

*Simulation study with gammapy v2 (dev), CTAO-South Prod5 IRFs. Companion to `UL-simulation-paper-script.ipynb`.*
*Code: `ul_simulation.py` (library), `ul_comparison_main.py` (paper figures), `ul_report.py` (this report: simulations + figures). Raw results: `report_results/*.ecsv`, `report_results/summary.json`.*

---

## 1. Summary

- **Both implementations behave as statistical theory predicts in the large-count regime.** At 100 h (33 500 background counts) the profile-likelihood and Bayesian upper limits (ULs) follow the Gaussian (Wilks) expectations realization by realization (Fig. 1, 2, 3). Median UL / σ is 1.54 (profile) and 1.97 (Bayes, prior ≥ 0), against 1.64 and 2.00 expected.
- **Coverage is respected.** All four methods give coverage consistent with the nominal 95 % at injected signals of 0, 1 and 3 σ, within the ±2.5 % binomial error of 100 realizations (Fig. 5). The Bayesian ULs with a positive prior over-cover near zero signal, as expected.
- **The Bayesian UL with a flat prior is always more conservative than the profile UL on the same data.** On the background-only Asimov dataset the ratio is 1.25 at 100 h, close to the Gaussian prediction of 1.21. It grows to 1.47 at 0.3 h and 1.62 at 0.03 h. The growth comes from the Poisson regime: the likelihood has a long upper tail, the posterior integrates it, and the profile only uses the likelihood up to Δ = 2.71.
- **The profile ULs scatter about twice as much as the Bayesian ones** (MAD/median 0.30–0.37 vs 0.16–0.22 for the constrained cases). They can also be very low after a downward background fluctuation, down to 0.5 σ_A. The Bayesian UL has a natural floor (≈ 2 σ at zero signal).
- **The "unconstrained" cases are really unconstrained only at 100 h.** The lower bounds are fixed amplitudes: −1e-14 for the profile norm and −3e-14 for the prior. In σ units they are −3.1 σ / −9.4 σ at 100 h, but only −0.11 σ / −0.33 σ at 0.3 h and −0.02 σ / −0.07 σ at 0.03 h. At low exposure the "negative" cases are therefore effectively constrained. This explains the constrained/unconstrained ratios ≈ 1 except at 100 h. The bounds should be defined in units of σ if the paper intends to compare "constrained vs unconstrained".
- **The nested sampling UL has a Monte Carlo noise of 3–7 % per run** (300 live points; Fig. 6). For a single free parameter, the posterior quantile can be computed exactly by grid integration (the new `"grid"` method). It agrees with the sampler without bias, to within 0.8 % on the median.
- **Scan-edge failures fixed.** At 0.03 h, 28 % (constrained) and 71 % (unconstrained) of the profiles do not reach Δ = 2.71 within the original norm scan. The automatic scan extension now recovers all of them (0 NaN).

---

## 2. Setup

| Item | Value |
|---|---|
| IRF | CTAO-South Prod5, 20 deg, `14MSTs37SSTs`, 180000 s |
| Region | ON circle r = 0.11 deg at 0.5 deg offset |
| Energy | reco 0.05–50 TeV, 6 bins/decade; fit range 0.05–50 TeV |
| Simulated sky | background only (power-law amplitude 1e-20); coverage runs inject 1 σ_A and 3 σ_A |
| Fit model | power law, index fixed at 2, **amplitude is the only free parameter** |
| Dataset | `SpectrumDataset` with known background (Cash statistic), type "on" |
| Exposures | 0.03 h, 0.3 h, 100 h → expected background in the ON region 10.0, 100.4, 33 482 counts |
| Realizations | 100 per exposure (seeds 0–99) + 1 Asimov; coverage: 100 per (exposure, injected level), seeds 1000–1099 |

Four test cases are run on **the same faked dataset** for each realization:

| Case | Method | Lower bound | Bound / σ_A (0.03 h, 0.3 h, 100 h) |
|---|---|---|---|
| profile-zero | profile likelihood, `stat_profile_ul_scipy`, Δ = 1.645² = 2.706 | norm ≥ 0 | 0, 0, 0 |
| profile-negative | same | norm ≥ −0.1 (amplitude ≥ −1e-14) | −0.02, −0.11, −3.13 |
| sampler-zero | UltraNest, 300 live points, uniform prior, 0.954 quantile | amplitude ≥ 0 | 0, 0, 0 |
| sampler-negative | same | amplitude ≥ −3e-14 | −0.07, −0.33, −9.40 |

Upper prior bound: 1e-11 (≈ 20 to 3 000 σ_A, irrelevant). For the Bayesian cases the figures use the exact `"grid"` posterior quantile, which is free of sampler noise. The nested-sampling values are compared to it in §5.4.

**Unit σ_A.** Everything is expressed in units of the Asimov uncertainty σ_A = UL_profile(Asimov) / √Δ, following Cowan et al. (2011). The values are σ_A = 4.6e-13, 9.1e-14 and 3.2e-15 cm⁻² s⁻¹ TeV⁻¹ at 0.03 h, 0.3 h and 100 h. In these units the Gaussian expectations do not depend on exposure, apart from the bounds.

---

## 3. Statistical expectations (Gaussian / Wilks regime)

Assume the log-likelihood in the amplitude μ is parabolic, $-2\ln L(\mu) = (\mu-\hat\mu)^2/\sigma^2 + \text{const}$, with $\hat\mu \sim \mathcal N(\mu_\text{true},\sigma)$. Write $z=\hat\mu/\sigma$ and $k=\sqrt{\Delta}=1.645$. All quantities below are in units of σ.

**Profile likelihood UL** (one-sided 95 %, $\Delta=2.706$), parameter bounded at $l$:

$$
\text{UL}_\text{prof}(z) =
\begin{cases} z + k & z \ge l \\ z + \sqrt{(l-z)^2 + k^2} & z < l \end{cases}
$$

- For $z \ge l$ this is the textbook $\hat\mu + 1.645\,\sigma$. It has exact 95 % coverage for all $\mu_\text{true}$ when $l=-\infty$.
- With $l = 0$ the fit sits at the boundary for half of the background-only realizations. The UL then decreases smoothly towards 0 as $z \to -\infty$ (Chernoff regime). Coverage becomes > 95 % near $\mu_\text{true}=0$, but the UL can become arbitrarily small. A "lucky" downward fluctuation of the background gives a UL much tighter than the experiment's sensitivity, which is the well-known motivation for CLs or power-constrained limits in HEP.
- On the Asimov dataset, $\text{UL}=1.645$ whatever the bound, so profile-zero and profile-negative must agree there.

**Bayesian UL**, flat prior on $[a, \infty)$, posterior quantile $c = 0.954$. The posterior is a truncated Gaussian, so

$$
\text{UL}_\text{Bayes}(z) = z + \Phi^{-1}\!\left[\Phi(a-z) + c\,\big(1-\Phi(a-z)\big)\right].
$$

- With $a=-\infty$: $\text{UL}= z + \Phi^{-1}(0.954) = z + 1.685$. A flat prior on an unbounded location parameter gives exact frequentist coverage, here 95.4 %.
- With $a=0$ and Asimov ($z=0$): $\text{UL}=\Phi^{-1}(0.977)=1.995$. **Expected Bayes/profile ratio: 1.995/1.645 = 1.21.**
- The Bayesian UL is bounded from below (→ 0 only very slowly), so its scatter is smaller than the profile's. It over-covers near the boundary.
- Note: 0.954 is the two-sided 2σ probability. A one-sided 95 % UL, matching Δ = 2.706, would use $c = 0.95$. With 0.954 the Bayesian ULs are 1.8 % (prior ≥ 0) to 2.4 % (unbounded) larger than with 0.95.

**Expected distributions under background only** ($z\sim\mathcal N(0,1)$, dashed lines in Fig. 3):

| | profile ≥ 0 | profile unbounded | Bayes ≥ 0 | Bayes unbounded |
|---|---|---|---|---|
| Asimov UL / σ | 1.645 | 1.645 | 1.995 | 1.685 |
| median UL / σ | 1.645 | 1.645 | 1.995 | 1.685 |
| coverage at μ_true = 0 | 100 % | 95 % | 100 % | 95.4 % |
| coverage at μ_true ≫ σ | 95 % | 95 % | 95.4 % | 95.4 % |

**Expected departures in the Poisson regime.** With 10–100 background counts spread over 18 energy bins, many bins have ≪ 1 expected count. The Cash likelihood in μ is then skewed: it is steeper towards negative μ, where the predicted counts approach 0, and has a long upper tail. Expected consequences:

1. The profile UL (a local quantity: where Δ reaches 2.71) is only mildly affected.
2. The Bayesian UL (an integral over the whole upper tail) increases.
3. The Bayes/profile ratio grows as counts decrease.

---

## 4. Proposed figures

1. **UL vs best fit** (Fig. 1): checks each method against its expected curve, realization by realization, and shows the effect of the bounds.
2. **Paired Bayes vs profile UL** (Fig. 2): one point per realization, with the same data for both methods. This removes the Poisson scatter and isolates the difference between the two methods.
3. **Cumulative UL distributions** (Fig. 3): the full shape of the distributions against the expected one. This is what a reader of a catalogue of ULs actually sees.
4. **Ratio and spread vs background counts** (Fig. 4): the transition from the Poisson to the Gaussian regime in one plot. This is the most compact figure for the paper.
5. **Coverage vs injected signal** (Fig. 5): the frequentist property that justifies calling the number a "95 % UL".
6. **Sampler Monte Carlo noise** (Fig. 6): the precision of a single nested-sampling UL compared with the exact posterior quantile.

The original per-realization scatter plot with side histogram (`plot_lc_triangle`, in `ul_comparison_main.py`) remains useful as an illustration. Figures 2 and 4 convey the comparison more directly.

---

## 5. Results: observed vs expected

### 5.1 UL vs best-fit amplitude

![Fig. 1](report_figures/fig1_ul_vs_bestfit.png)

*Fig. 1. UL / σ_A vs maximum-likelihood amplitude of the profile-negative fit (same seed) / σ_A. Lines show the Gaussian expectation of §3 for each case with its own bound. The dotted vertical line is the profile-negative lower bound.*

- **100 h:** the points sit on the expected curves for all four cases. This includes the boundary regime of profile-zero for $\hat\mu<0$ (orange points bending down) and the Bayes ≥ 0 floor (blue). It validates the gammapy profile and sampler implementations and the identification of the amplitude column in the samples.
- **0.3 h and 0.03 h:** the best fit piles up at the lower bound, which is at −0.11 σ_A and −0.02 σ_A. The "unconstrained" fit cannot go further negative, so all downward fluctuations are compressed onto the bound line. For positive fluctuations both methods lie above the Gaussian curve, and the Bayesian ULs more so. This is the asymmetric Poisson likelihood of §3.

### 5.2 Bayesian vs profile UL on the same data

![Fig. 2](report_figures/fig2_paired_bayes_vs_profile.png)

*Fig. 2. Bayesian UL vs profile UL for the same realization. Lines show the expected relation (parametric in z); the dashed line is 1:1.*

- The Bayesian UL is larger than the profile UL in **100 %** of realizations at 0.03 h and 0.3 h (both priors) and at 100 h with the ≥ 0 prior. With the unbounded prior at 100 h it is larger in 95 %, as the Gaussian expectation says: for $z \gtrsim 0$ the two ULs differ only by 1.685 vs 1.645.
- The median per-realization ratio Bayes/profile:

| exposure | N_bkg | ratio, prior ≥ 0 | ratio, "negative" | Gaussian expectation (Asimov), ≥ 0 / negative |
|---|---|---|---|---|
| 0.03 h | 10 | 1.89 (1.14–2.23) | 1.87 (1.13–2.21) | 1.21 / 1.20 |
| 0.3 h | 100 | 1.56 (1.09–2.20) | 1.47 (1.09–2.06) | 1.21 / 1.16 |
| 100 h | 33 482 | 1.27 (1.02–2.06) | 1.05 (0.75–2.71) | 1.21 / 1.02 |

- The largest ratios occur where the profile UL is smallest, i.e. after a downward fluctuation. There the profile UL drops while the Bayesian UL stays near its floor.

### 5.3 Distributions and spread

![Fig. 3](report_figures/fig3_ecdf.png)

*Fig. 3. Cumulative distribution of UL / σ_A over 100 background-only realizations (solid) vs the Gaussian expectation (dashed).*

![Fig. 4](report_figures/fig4_ratio_vs_background.png)

*Fig. 4. Left: Bayesian/profile UL ratio vs background counts: median of the per-realization ratio, Asimov ratio, and Gaussian expectation. Right: relative spread (MAD/median) of the ULs.*

| exposure | case | median UL/σ_A (expected) | Asimov UL/σ_A (expected) | MAD/median |
|---|---|---|---|---|
| 0.03 h | profile ≥ 0 | 1.33 (1.64) | 1.65 (1.65) | 0.30 |
| | profile "neg." | 1.31 (1.64) | 1.64 (1.65) | 0.30 |
| | Bayes ≥ 0 | 2.50 (2.00) | 2.66 (2.00) | 0.16 |
| | Bayes "neg." | 2.45 (1.97) | 2.62 (1.97) | 0.17 |
| 0.3 h | profile ≥ 0 | 1.52 (1.64) | 1.65 (1.65) | 0.36 |
| | profile "neg." | 1.48 (1.64) | 1.64 (1.65) | 0.40 |
| | Bayes ≥ 0 | 2.33 (2.00) | 2.42 (2.00) | 0.18 |
| | Bayes "neg." | 2.19 (1.90) | 2.32 (1.90) | 0.23 |
| 100 h | profile ≥ 0 | 1.54 (1.64) | 1.65 (1.65) | 0.37 |
| | profile "neg." | 1.54 (1.64) | 1.64 (1.65) | 0.46 |
| | Bayes ≥ 0 | 1.97 (2.00) | 2.05 (2.00) | 0.22 |
| | Bayes "neg." | 1.64 (1.68) | 1.74 (1.68) | 0.43 |

*The binomial uncertainty on a median from 100 realizations is ≈ ±0.1 σ_A.*

Observations:

- **Large counts (100 h):** every observed median and Asimov value agrees with the Gaussian expectation to within the statistical precision.
- **Profile, low counts:** the medians are lower than expected (1.33 at 0.03 h). Many realizations have a downward fluctuation and the fit sits at the (≈ 0) bound, where the UL is below 1.645 σ_A (Fig. 1). The Asimov value stays exactly 1.645, as expected.
- **Bayes, low counts:** the medians are 17–25 % above the Gaussian expectation, because of the heavy upper tail of the Poisson likelihood.
- **Spread:** the profile ULs scatter about twice as much as the Bayes ≥ 0 ULs at every exposure. This is the floor effect described in §3. The Bayes "negative" spread grows with exposure (0.17 → 0.43) because the prior becomes truly unbounded and the UL then tracks $\hat\mu$ one-to-one.
- **Constrained / unconstrained** (median UL ratio):
  - Profile: 1.01, 1.03, 1.00 at 0.03 h, 0.3 h, 100 h.
  - Bayes: 1.02, 1.06, 1.21.
  - At low exposure the "negative" bounds are ≈ 0 in σ units, so no difference is expected. At 100 h the expected Bayes value is 2.00/1.68 = 1.19 (observed 1.21). For the profile the expected value is 1.00: the bound only matters for $\hat\mu<0$, and there the profile-zero UL is *larger* than the unbounded one but both are below the median.

### 5.4 Coverage

![Fig. 5](report_figures/fig5_coverage.png)

*Fig. 5. Fraction of realizations with UL ≥ injected amplitude (100 realizations per point, binomial errors). Lines: Gaussian expectation for each case with its bounds.*

| exposure | injected / σ_A | profile ≥ 0 | profile "neg." | Bayes ≥ 0 | Bayes "neg." |
|---|---|---|---|---|---|
| 0.3 h | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| | 1 | 0.98 | 0.96 | 1.00 | 0.99 |
| | 3 | 0.94 | 0.94 | 0.98 | 0.97 |
| 100 h | 0 | 1.00 | **0.95** | 1.00 | **0.95** |
| | 1 | 0.93 | 0.93 | 1.00 | 0.93 |
| | 3 | 0.93 | 0.93 | 0.93 | 0.93 |

- All values agree with the expectations within ≈ 1 σ (±0.022–0.026 at 95 %). Neither method shows significant under-coverage. Confirming the slight deficit at 100 h (0.93) would need ~1 000 realizations per point.
- At 100 h with zero injected signal, the unbounded methods give **5 % of ULs below the true value (≈ 0)**, i.e. negative or near-zero ULs. This is the exact frequentist behaviour, but such ULs are physically meaningless; it is the classic argument for bounded or Feldman–Cousins intervals.
- The over-coverage of Bayes ≥ 0 near zero signal (1.00 at 1 σ, expected 0.995–1.00) is the price paid for its stability.
- Coverage was not measured at 0.03 h. It is the regime where Wilks' theorem is least reliable, and checking it is a recommended follow-up (§7).

### 5.5 Precision of nested sampling

![Fig. 6](report_figures/fig6_sampler_noise.png)

*Fig. 6. 30 independent UltraNest runs (300 live points) on the same dataset (seed 0), relative to the exact posterior quantile from grid integration.*

| exposure | prior ≥ 0: std (mean) | prior "neg.": std (mean) |
|---|---|---|
| 0.3 h | 3.0 % (+0.1 %) | 3.3 % (−0.1 %) |
| 100 h | 3.5 % (+0.6 %) | 7.2 % (−0.8 %) |

- Over the 100-realization main run, nested sampling vs grid: median offset −0.6 % to +0.7 %, scatter 2.6–3.7 %; 7.8 % for the unbounded prior at 100 h.
- The sampler is unbiased, but a single run carries a 3–7 % Monte Carlo error on the UL.
- The error is larger when the prior volume is huge compared with the posterior (prior "neg." at 100 h: the posterior occupies ~10⁻³ of [−3e-14, 1e-11]).
- The error is subdominant to the 20–45 % realization-to-realization scatter, but not negligible for Asimov-based "expected UL" numbers or for ratios quoted to two decimals (e.g. the Asimov ratios in the notebook).
- Increasing `min_num_live_points`, or using the exact grid integration when only one parameter is free, removes it.

### 5.6 Spread of the ULs at zero signal (Brazil bands)

The "Brazil bands" are the expected-limit bands of Cowan et al. (2011): the median and the ±1σ / ±2σ quantiles (2.3, 15.9, 50, 84.1, 97.7 %) of the UL distribution under the background-only hypothesis. They were computed in three ways for 0.03 h (10 background counts, Poisson regime) and 100 h (33 500 counts, Gaussian regime):

1. **Expected (Asimov, Gaussian limit).** The background-only Asimov dataset gives σ_A. Assuming $\hat\mu/\sigma_A = z \sim \mathcal N(0,1)$ and using that the UL is a monotonic function of $z$ (§3), the quantile of the UL at $\Phi(N)$ is $\mathrm{UL}(z=N)$. For the unbounded profile this is Cowan's $\mathrm{UL}_N = \sigma_A(1.645+N)$; the bounded and Bayesian versions use the formulas of §3. In units of σ_A the whole band, not only the median, follows from Asimov.
2. **Shifted Asimov.** UL computed on Asimov datasets with an injected amplitude $N\sigma_A$, $N = -2 \dots 2$. This uses the true Poisson likelihood shape around a best fit at $N\sigma_A$ (no parabolic approximation), but still assumes $\hat\mu$ is Gaussian with width σ_A. No bin had negative expected counts.
3. **Observed.** Empirical quantiles of 1 000 background-only realizations per exposure (seeds 2000–2999, independent of §5.1–5.4), with bootstrap uncertainties. Bayesian ULs use the exact grid quantile.

![Fig. 7](report_figures/fig7_brazil_bands.png)

*Fig. 7. For each case: expected Brazil band (left box: green ±1σ, yellow ±2σ, dashed median) with the shifted-Asimov ULs (diamonds), and the observed quantiles of 1 000 realizations (right box, case colour: dark 15.9–84.1 %, light 2.3–97.7 %, black line median, error bars = bootstrap uncertainty of each quantile). Star: background-only Asimov UL. The numbers give the fraction of observed ULs inside the expected ±1σ and ±2σ bands (nominal 0.683 and 0.954). At 0.03 h the "unconstrained" cases are bounded at −0.02 σ_A (profile) and −0.07 σ_A (Bayes), so they are effectively constrained (§2).*

**UL / σ_A at the quantiles −2σ, −1σ, median, +1σ, +2σ** (observed ± bootstrap error / expected Gaussian / shifted Asimov):

| exposure | case | −2σ | −1σ | median | +1σ | +2σ |
|---|---|---|---|---|---|---|
| 0.03 h | profile ≥ 0 | 0.73±0.01 / 0.59 / 0.87 | 0.93±0.01 / 0.93 / 1.02 | 1.43±0.03 / 1.65 / 1.65 | 2.66±0.09 / 2.65 / 3.29 | 4.71±0.24 / 3.65 / 4.67 |
| | profile "neg." | 0.71±0.01 / 0.57 / 0.85 | 0.91±0.01 / 0.91 / 1.00 | 1.42±0.04 / 1.65 / 1.64 | 2.66±0.10 / 2.65 / 3.29 | 4.71±0.24 / 3.65 / 4.67 |
| | Bayes ≥ 0 | 1.63±0.02 / 1.08 / 1.88 | 1.98±0.01 / 1.44 / 2.11 | 2.55±0.02 / 2.00 / 2.66 | 3.51±0.06 / 2.77 / 3.94 | 5.30±0.22 / 3.70 / 5.23 |
| | Bayes "neg." | 1.57±0.02 / 1.03 / 1.82 | 1.92±0.02 / 1.41 / 2.05 | 2.51±0.03 / 1.97 / 2.62 | 3.49±0.06 / 2.76 / 3.94 | 5.30±0.22 / 3.69 / 5.23 |
| 100 h | profile ≥ 0 | 0.57±0.02 / 0.59 / 0.56 | 0.91±0.02 / 0.93 / 0.89 | 1.62±0.03 / 1.65 / 1.65 | 2.65±0.05 / 2.65 / 2.71 | 3.67±0.10 / 3.65 / 3.76 |
| | profile "neg." | −0.45±0.09 / −0.35 / −0.47 | 0.56±0.05 / 0.65 / 0.56 | 1.62±0.04 / 1.65 / 1.64 | 2.65±0.05 / 2.65 / 2.71 | 3.67±0.10 / 3.65 / 3.76 |
| | Bayes ≥ 0 | 1.07±0.03 / 1.08 / 1.06 | 1.46±0.02 / 1.44 / 1.45 | 2.04±0.02 / 2.00 / 2.05 | 2.84±0.04 / 2.77 / 2.88 | 3.78±0.10 / 3.70 / 3.86 |
| | Bayes "neg." | −0.37±0.09 / −0.32 / −0.40 | 0.68±0.05 / 0.68 / 0.67 | 1.72±0.04 / 1.68 / 1.74 | 2.74±0.05 / 2.68 / 2.80 | 3.77±0.10 / 3.68 / 3.85 |

**Fraction of observed ULs inside the expected bands** (binomial error ±0.015 for ±1σ, ±0.007 for ±2σ):

| exposure | case | in ±1σ (0.683) | in ±2σ (0.954) | below −2σ / above +2σ (0.023 each) |
|---|---|---|---|---|
| 0.03 h | profile ≥ 0 | 0.687 | 0.935 | 0.000 / 0.065 |
| | profile "neg." | 0.679 | 0.935 | 0.000 / 0.065 |
| | Bayes ≥ 0 | 0.600 | 0.871 | 0.000 / 0.129 |
| | Bayes "neg." | 0.612 | 0.875 | 0.000 / 0.125 |
| 100 h | profile ≥ 0 | 0.665 | 0.946 | 0.028 / 0.026 |
| | profile "neg." | 0.659 | 0.945 | 0.029 / 0.026 |
| | Bayes ≥ 0 | 0.673 | 0.947 | 0.025 / 0.028 |
| | Bayes "neg." | 0.662 | 0.945 | 0.027 / 0.028 |

Interpretation:

- **100 h (Gaussian regime): the Asimov bands describe the ULs.** Every observed quantile agrees with the Gaussian expectation to within ≈ 1–2 bootstrap errors (≤ 0.1 σ_A), for both methods and both bounds. The band fractions (0.66–0.67 and 0.945–0.947) are within ≈ 1.5 binomial errors of 0.683 and 0.954, and the tails are symmetric (2.5–2.9 % on each side). The shifted-Asimov points are equally good; they lie slightly above at +2σ (3.76–3.86 vs 3.67–3.78) because the likelihood widens with the counts of the injected signal.
- **100 h, unconstrained:** the −2σ quantile is negative for both methods (−0.45 σ_A profile, −0.37 σ_A Bayes). 5.8 % (profile) and 4.7 % (Bayes) of the ULs are below zero, against ≈ 5 % expected. The constraint only changes the lower half of the band: it lifts the −2σ edge from −0.45 to 0.57 (profile) and from −0.37 to 1.07 σ_A (Bayes), For the profile it leaves the median and upper edges unchanged. For Bayes the prior truncation also raises the median (1.72 → 2.04 σ_A) and, slightly, the upper edges (+0.1 σ_A).
- **0.03 h (Poisson regime): the Gaussian Asimov bands do not describe the ULs.** The distribution is skewed: compressed at low values and with a long upper tail. For the **profile**, the ±1σ edges agree by coincidence (0.93 and 2.66, so 69 % fall inside the expected ±1σ band). However, the median is 13 % low (1.43 vs 1.645, Asimov) and the +2σ edge is 29 % high (4.71 vs 3.65). There are no ULs below the expected −2σ edge, and 6.5 % are above the expected +2σ edge. For **Bayes**, the whole band is shifted upwards by 27–52 %. Only 60 % / 87 % of the ULs are inside the expected ±1σ / ±2σ bands, and 13 % are above +2σ.
- **The shifted Asimov reproduces the upper tail at 0.03 h:** +2σ edge 4.67 vs 4.71 observed (profile), 5.23 vs 5.30 (Bayes). This is the main gain over the parabolic Gaussian formula. It captures the wider Poisson likelihood for upward fluctuations. It is less good elsewhere. It overestimates the +1σ edge (3.29 vs 2.66 profile) and the lower edges (0.87 vs 0.73), and the median stays at the Asimov value (1.65) instead of the observed 1.43. These quantiles are controlled by the non-Gaussian, discrete distribution of $\hat\mu$ for 10 counts (≈ 58 % of the realizations have $n \le b$ and fit at the bound), which no single Asimov dataset can represent. Only toy Monte Carlo gives the correct band there.
- **Constrained vs unconstrained at 0.03 h:** the rows are identical within 0.06 σ_A, as expected from bounds at −0.02 σ_A (profile) and −0.07 σ_A (Bayes). The "unconstrained" label is meaningless at this exposure (see §1, §2).
- **Profile vs Bayes:** relative to its median, the lower half of the Bayesian band (−2σ to median) is narrower than the profile one: 48 % vs 65 % of the median at 100 h, and 36 % vs 49 % at 0.03 h (constrained cases). This is the Bayesian floor. In the Poisson regime the Bayesian band is also shifted upwards (median 2.55 vs 1.43 σ_A), consistent with §5.3.
- **Practical consequence:** an "expected UL" or sensitivity band computed from Asimov (σ_A and the Gaussian formulas, as is common practice) is reliable at high counts. At ~10 background counts it underestimates the upper tail of the profile ULs (and the whole Bayes band). There, toy Monte Carlo bands (or at least shifted-Asimov bands for the upper edges) should be used.

---

## 6. Pros and cons for faint-source ULs with IACTs / CTAO

| | Profile likelihood (frequentist) | Bayesian (nested sampling) |
|---|---|---|
| **Statistical guarantee** | Coverage ≈ 95 % by construction (Wilks), checked here at 100 and 10⁴ counts. Directly comparable with published IACT and Fermi-LAT ULs, which mostly use Δ = 2.71 profile or Rolke / Feldman–Cousins. | Credible interval: answers "where is the flux, given the data and my prior". No coverage guarantee in general, but over-covers near zero with a positive flat prior, as seen here. |
| **Behaviour at zero / negative excess** | With a physical bound the UL can become very small after a downward fluctuation (0.5 σ_A here); with no bound it can be negative (5 % at 100 h). Both are hard to interpret and motivate CLs or power-constrained limits. | Natural floor near 2 σ: the UL never claims more than the instrument can do. Fewer "too good" limits, 2× smaller scatter (MAD/median 0.16–0.22 vs 0.30–0.46). |
| **Low counts (short exposure, high energy, small ROI)** | Local criterion (Δ = 2.71) on a skewed likelihood; the Asimov UL stays at 1.645 σ_A. Needs a wide enough scan, and failed in 28–71 % of cases at 0.03 h before the automatic extension. | Integrates the full Poisson tail, giving larger ULs: 1.6–1.9× the profile at 10 background counts vs 1.2× expected in the Gaussian limit. Arguably the more honest number at very low counts, but strongly prior-dependent there. |
| **Prior dependence** | None (but the choice of bound matters at low exposure, and should be set in units of σ). | Central. Uniform in amplitude favours large fluxes; a log-uniform or Jeffreys prior would lower the ULs, most strongly in the Poisson regime. The prior bounds used here (−3e-14) mean different things at different exposures. Must be stated explicitly in a paper. |
| **Nuisance parameters** (spectral index, background normalisation, IRF systematics) | Profiled: a maximisation per scan point. Can under-cover when the nuisance parameters are poorly constrained. | Marginalised. Naturally includes uncertainties with priors (e.g. a Gaussian systematic on effective area). This is the main practical advantage once more than one parameter is free. |
| **Cost** (measured, 1 core) | 0.8–0.9 s per UL (FluxPointsEstimator scan). Deterministic. Trivial to run for catalogues of thousands of positions or energy bins. | 3.4 s (0.3 h) to 5.6 s (100 h) per UL with 300 live points, growing with the number of free parameters. 3–7 % Monte Carlo noise unless the live points are increased. For one parameter, the exact grid integration takes 1.0 s and is deterministic. |
| **Combination / population studies** | Likelihood profiles can be stacked (joint likelihood). | Posteriors combine naturally in hierarchical (population) models, e.g. stacking non-detected sources or dark-matter J-factor marginalisation. |
| **Reporting** | One number + Δ convention. Easy to reproduce. | Requires prior, quantile and sampler settings. Posterior samples carry more information (shape, multimodality). |

**Recommendations for the paper / use case**

1. For **catalogue-like or survey ULs** (many positions or energy bins, comparison with literature), use the **profile likelihood with a physical bound (≥ 0)**. Quote the sensitivity (or Asimov UL) next to each UL, so that "lucky" low limits are recognisable, or adopt a power-constrained limit (UL = max(UL, sensitivity)).
2. For **single deep targets with poorly constrained nuisance parameters**, or for population and dark-matter analyses, the **Bayesian approach** is preferable. State the prior explicitly and show the prior sensitivity (uniform vs log-uniform).
3. **Use consistent confidence levels:** 0.95 quantile with Δ = 2.706. The current 0.954 vs 95 % mismatch inflates the Bayesian ULs by ≈ 2 %.
4. **Define "unconstrained" bounds in units of σ_A** (e.g. −5 σ_A) rather than as fixed amplitudes, so that the comparison means the same thing at all exposures.
5. For **one free parameter**, compute the Bayesian UL with the exact grid integration, or check the sampler against it. When the sampler is needed, raise the live points until the quantile error is ≪ the effect being studied.

---

## 7. Limitations and possible follow-ups

- 100 realizations per configuration gives ±0.1 σ_A on medians and ±2.5 % on coverage. Use ≥ 1 000 realizations for paper-quality coverage curves; the computation is parallel and takes ≈ 30 min for 1 000 realizations of 4 cases on 10 cores.
- Coverage at 0.03 h (10 background counts), where the profile is most likely to deviate from Wilks.
- Only the amplitude is free. Freeing the spectral index or a background normalisation (`SpectrumDatasetOnOff`, `dataset_type="onoff"`, already supported) would test the nuisance-parameter behaviour, which is where the two approaches differ most.
- Prior sensitivity of the Bayesian UL: log-uniform and Jeffreys priors.
- Feldman–Cousins / CLs comparison as the other standard frequentist choices for bounded parameters.
- The `norm_ul` returned by `FluxPointsEstimator` is NaN when the constrained best fit is at the boundary; a gammapy PR is in progress, so it is not used here.

---

## 8. Reproducibility

```
python ul_report.py            # reruns or reloads report_results/*.ecsv and writes report_figures/*.png
```

- Main run: 303 realizations, 468 s.
- Coverage: 400 realizations, 655 s.
- Sampler noise: 60 realizations, 91 s.
- Brazil bands (§5.6): `python brazil_bands.py`, 2 002 realizations (1 000 per exposure + Asimov, profile + grid Bayes, no nested sampling), 925 s; results in `report_results/uls_brazil.ecsv` and `report_results/brazil_summary.json`.
- All on 10 cores, gammapy dev (`2.0.dev3770`), UltraNest 4.4.0.
- Set `recompute = True` in `ul_report.py` to rerun the simulations.
- Seeds are fixed (`seed="range"`); the sampler itself is not seeded, so nested-sampling ULs reproduce only to within the ≈ 3 % of §5.5.
