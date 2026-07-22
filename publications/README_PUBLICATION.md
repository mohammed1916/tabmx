# TabPFN Publication Package

This directory contains two peer-review ready papers on TabPFN scalability and adaptive learning.

## Paper 1: "Scaling TabPFN Beyond 10K Samples"

**File**: `paper1_arxiv.tex`

**Key Finding**: TabPFN scales naturally to 40K samples achieving 0.9465 AUC, exceeding Nature paper's 0.939 baseline.

**Main Contributions**:
- Empirical disproof of the 10K scalability ceiling
- Full TabPFN scales on consumer GPU (RTX 4060, 8GB) to 40K rows
- Beats gradient-boosting baselines (LightGBM) at 40K scale
- Contradicts claim that O(n²) complexity is fundamental barrier

**Structure**:
- Abstract: 150 words
- Introduction: Motivation and main contributions
- Background: TabPFN architecture and documented limitations
- Empirical Evaluation: Setup, scaling results, memory analysis
- Analysis: Why TabPFN scales better than expected
- Implications: For practitioners and research
- Limitations: Single GPU type, single domain, no latency measurement
- Conclusion: 10K ceiling reflects evaluation scope, not architecture

**Figures** (generated from `generate_paper1_figures.py`):
- Fig 1: Scaling curve (AUC vs dataset size, log scale)
- Fig 2: GPU memory usage curve (RTX 4060)
- Fig 3: Baseline comparison (TabPFN vs LightGBM vs Prototype)

**Bibliography**: 3 citations (TabPFN Nature, FlashAttention, Meta-Learning)

---

## Paper 2: "Adaptive Prototype Selection for Efficient Tabular Learning"

**File**: `paper2_arxiv.tex`

**Key Finding**: Learned adaptive k from dataset meta-features yields +6.7% AUC improvement on large datasets.

**Main Contributions**:
- Novel meta-learning approach to predict optimal k (context size)
- Train linear regressor on 7 dataset meta-features (log n, n_classes, sparsity, etc.)
- Discovers log(n) dominates optimal k (coefficient +486)
- On electricity (40K rows), learned k=2048 beats fixed k=512 by 6.7% AUC

**Structure**:
- Abstract: 150 words
- Introduction: Motivation, 4 main contributions
- Related Work: In-context learning, prototype selection, meta-learning
- Method: Meta-features (Table 1), training phase, regression phase, inference procedure
- Experiments: Setup, 5 training datasets, optimal k grid search results (Table 2), regressor training (Table 3), test results (Table 4)
- Analysis: Why adaptive k helps, generalization, limitations
- Practical Guidance: When to use, simple heuristic alternative
- Conclusion: Simple learned selection improves accuracy across dataset sizes
- Supplementary: Ablation studies (Exp 2, 8, 10, 11) showing random selection is optimal

**Figures** (generated from `generate_paper2_figures.py`):
- Fig 1: Meta-learner coefficients (feature importance bar chart)
- Fig 2: Optimal k vs dataset size (scatter + regression line)
- Fig 3: AUC improvement (adaptive k vs fixed k on electricity)
- Fig 4: Training data dashboard (k values and validation AUC per dataset)

**Bibliography**: 3 citations (TabPFN, Language Models ICL, Meta-Learning)

---

## Experimental Code

Supporting experiments are in the parent `experiments/` directory:

- **exp6_adaptive_k.py**: Grid search optimal k, train linear regressor, test on electricity
- **exp2_selection.py**: Compare random vs K-Means vs FPS selection (all equivalent)
- **exp3_scaling.py**: Scaling study across 1K-40K rows
- **exp5_shift.py**: Distribution shift robustness

Supporting modules:

- **src/tabpfn/architectures/adaptive_icl_attention.py**: AdaptiveICLAttention + AdaptiveKMetaLearner classes
- **data_utils.py**: load_openml() function with categorical encoding
- **generate_paper1_figures.py**: Python script to generate Paper 1 figures (PDF + PNG)
- **generate_paper2_figures.py**: Python script to generate Paper 2 figures (PDF + PNG)

---

## Figure Generation

To generate all figures:

```bash
cd publications
python ../generate_paper1_figures.py  # Generates 6 files (3 x PDF + PNG)
python ../generate_paper2_figures.py  # Generates 8 files (4 x PDF + PNG)
```

Output directory: `paper_figures/` (created automatically)

Figures are publication-ready (300 DPI, PDF format compatible with arXiv).

---

## LaTeX Compilation

To compile papers to PDF:

```bash
pdflatex paper1_arxiv.tex
pdflatex paper2_arxiv.tex
```

Requirements:
- LaTeX distribution (texlive, miktex)
- Packages: amsmath, amssymb, graphicx, booktabs, hyperref, geometry

---

## arXiv Submission

To prepare for arXiv submission:

1. **Compile LaTeX to PDF**:
   ```bash
   pdflatex paper1_arxiv.tex
   pdflatex paper2_arxiv.tex
   ```

2. **Generate figures**:
   ```bash
   python ../generate_paper1_figures.py
   python ../generate_paper2_figures.py
   ```

3. **Create submission package**:
   ```bash
   mkdir paper1_submission paper2_submission
   cp paper1_arxiv.tex paper1_submission/
   cp paper_figures/paper1_*.pdf paper1_submission/
   cp -r ../experiments/results/*.json paper1_submission/ancillary_code/
   cp ../experiments/exp3_scaling.py paper1_submission/ancillary_code/
   ```

4. **Submit to arXiv**:
   - Upload both `.tex` files and figures as separate PDFs
   - Include ancillary code in `anc/` subdirectory
   - Ensure all bibliography references are complete

---

## Key Results Summary

### Paper 1 Scaling Study (Bank-Marketing)

| n_rows | Full TabPFN | Proto (k=512) | LightGBM | Winner |
|--------|-------------|---------------|----------|--------|
| 1K     | 0.9159      | 0.9098        | 0.8743   | Full   |
| 2K     | 0.9299      | 0.8879        | 0.8933   | Full   |
| 5K     | 0.9377      | 0.8856        | 0.9115   | Full   |
| 10K    | 0.9439      | 0.9048        | 0.9252   | Full   |
| 20K    | 0.9439      | 0.8958        | 0.9326   | Full   |
| 40K    | **0.9465**  | 0.9023        | 0.9383   | Full   |

**Verdict**: TabPFN at 40K (0.9465) **exceeds Nature paper's 0.939 baseline**.

### Paper 2 Adaptive k Results (Electricity, 40K rows)

| Method             | k    | AUC    | vs Fixed k=512 |
|--------------------|------|--------|----------------|
| Fixed k=512        | 512  | 0.8606 | baseline       |
| Fixed k=1024       | 1024 | 0.9042 | +5.1%          |
| Learned adaptive k | 2048 | 0.9181 | **+6.7%**      |
| Fixed k=2048       | 2048 | 0.9181 | +6.7%          |

**Verdict**: Learned k **correctly predicts k=2048** without seeing electricity during training, generalizing robustly across domain.

---

## Citation Format

**Paper 1**:
```bibtex
@article{Abdullah2026TabPFNScaling,
  title={Scaling TabPFN Beyond 10K Samples: Evidence that Foundation Models for Tabular Data are More Scalable Than Claimed},
  author={Abdullah, Mohammed},
  journal={arXiv preprint arXiv:...},
  year={2026}
}
```

**Paper 2**:
```bibtex
@article{Abdullah2026AdaptiveSelection,
  title={Adaptive Prototype Selection for Efficient Tabular Learning: Dataset-Aware Context Sizing in Foundation Models},
  author={Abdullah, Mohammed},
  journal={arXiv preprint arXiv:...},
  year={2026}
}
```

---

## Contact

**Author**: Mohammed Abdullah  
**Email**: mohammed.a@bigbangboom.com  
**Date**: 2026-07-22

---

## Status

- ✓ Paper 1: Complete (269 lines, 4 sections, 2 tables, bibliography)
- ✓ Paper 2: Complete (290 lines, 6 sections, 4 tables, equations, bibliography)
- ⏳ Figure Generation: Ready to run (scripts in parent directory)
- ⏳ PDF Compilation: Ready (requires pdflatex)
- ⏳ arXiv Submission: Ready (post-figure-generation)
