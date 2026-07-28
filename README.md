# Context Sampling in TabPFN — Research Fork

This repository is a **research fork** of [TabPFN](https://github.com/PriorLabs/TabPFN)
(Prior Labs). The upstream library is used unmodified as a dependency; our
contribution is an empirical study of **how the in-context training set
(the "context") should be chosen** when using TabPFN on small tabular datasets.

The upstream project's README is preserved at
[README_TabPFN.md](README_TabPFN.md), and its license and attribution are
unchanged (see [License](#license)).

---

## What this fork adds

TabPFN predicts by conditioning on a set of labeled training rows supplied as
context. Two choices are left to the practitioner: **how many** rows go into the
context ($k$), and **which** rows. We study how much those choices matter and,
where they do, *why*. The work is organized as two papers and a reproducible
experiment suite.

### Paper 2 (main study) — *Understanding Context Sampling in TabPFN on Small Tabular Datasets*

Fifteen small OpenML datasets, GPU (RTX 4060). Three linked questions:

- **H1 — Stability.** Different random contexts of the same size give different
  predictions, and larger contexts are more stable: the coefficient of variation
  of AUC falls from roughly 6–18% at $k{=}16$ to 1–4% at the largest $k$ on
  datasets with headroom. Context size is a *reliability* knob, not just an
  accuracy knob.
- **H2 — What makes a context good?** Accuracy *correlates* with how well a
  random context matches the training distribution — but a **controlled
  experiment overturns the causal reading**. When we construct contexts that
  match the feature means closely, accuracy drops (by up to 0.5 AUC), because
  that construction also destroys diversity. Disentangling the two factors, a
  mixed-effects model finds **diversity/coverage** predicts accuracy
  ($\beta{=}{+}0.23$, $p{=}3\times10^{-12}$) while feature-mean shift does not
  ($\beta{=}{-}0.01$, $p{=}0.71$). The per-dataset picture is mixed, so we state
  this as a directionally dominant aggregate result, not a universal law.
- **H3 — Selection cost.** K-Means and farthest-point selection match uniform
  random to within ~1 AUC point while costing two to three orders of magnitude
  more to compute. They already maximize diversity — the property that matters —
  so random (which supplies comparable coverage for free) is a strong, cheap
  default.

Paper source: [`publications/paper2/paper2_context_sampling.tex`](publications/paper2/paper2_context_sampling.tex).

### Paper 1 (empirical report) — *Context Size and Prototype Selection for TabPFN*

A broader, more descriptive report: full-context accuracy vs. training-set size
(diminishing returns), an AUC-vs-$k$ sweep, prototype-selection comparisons, and
distribution-shift stress tests. Source:
[`publications/paper1/paper_arxiv.tex`](publications/paper1/paper_arxiv.tex).

> Every number in both papers is read directly from logged experiment outputs.
> We report only measured results, with effect sizes and confidence intervals.

---

## Repository layout (our additions)

```
experiments/
  exp14_context_sampling.py   # H1 stability, H2 correlation, H3 selection, stress tests
  exp15_controlled.py         # controlled representativeness + extended blocks
  exp16_disentangle.py        # shift-vs-diversity disentangling (6 constructions)
  analyze_exp14.py            # CV, correlations, selection summaries
  analyze_exp15.py            # controlled effect sizes (Cohen's d, Cliff's delta)
  analyze_exp16.py            # partial correlations, standardized betas
  mixed_effects_exp16.py      # AUC ~ diversity + shift + (1|dataset)
  data_utils.py               # OpenML loading with categorical handling
  results/                    # raw JSON outputs (exp14/15/16 + analysis summaries)
publications/
  paper1/                     # paper_arxiv.tex + generate_figures.py + figures/
  paper2/                     # paper2_context_sampling.tex + generate_paper2_figures.py + figures/
TRACKER.md                    # full research log (decisions, dead ends, results)
```

Everything under `src/`, `tests/`, `scripts/`, etc. is upstream TabPFN.

---

## Reproducing the study

Requires a CUDA GPU (the study used an RTX 4060). Install TabPFN and the analysis
dependencies, then run the experiments — each writes timestamped JSON to
`experiments/results/`:

```bash
pip install tabpfn scikit-learn scipy statsmodels matplotlib
cd experiments

python exp14_context_sampling.py     # H1 / H2-correlation / H3 / stress tests
python exp15_controlled.py           # controlled representativeness
python exp16_disentangle.py          # disentangling shift vs diversity
```

Then regenerate the analysis summaries, statistics, and figures:

```bash
python analyze_exp14.py results/exp14_context_sampling_*.json
python analyze_exp15.py results/exp15_controlled_*.json
python analyze_exp16.py results/exp16_disentangle_*.json
python mixed_effects_exp16.py results/exp16_disentangle_*.json

cd ../publications/paper2 && python generate_paper2_figures.py
cd ../paper1 && python generate_figures.py
```

All datasets are public OpenML tasks; dataset IDs, seeds, and splits are recorded
in the scripts and result files. Because the raw JSON outputs are committed under
`experiments/results/`, the tables and figures can also be regenerated without
re-running the (GPU-bound) experiments.

---

## License

This fork inherits the upstream licenses. The TabPFN code and model weights are
governed by the Prior Labs License and the model licenses described in the
[upstream README](README_TabPFN.md#license); our added research code
(`experiments/`, `publications/`) is provided for reproducibility of the study
above. TabPFN is © Prior Labs GmbH.

## Citing the underlying model

If you use TabPFN itself, please cite the original work (BibTeX in
[README_TabPFN.md](README_TabPFN.md#citation)):

> Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M.,
> Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on
> small data with a tabular foundation model. *Nature*, 637(8045), 319–326.
