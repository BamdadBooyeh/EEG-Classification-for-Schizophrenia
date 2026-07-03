
# EEG-Based Schizophrenia Detection: Replication & Critical Evaluation

> **Thesis:** *Feature Complexity vs Sample Size in EEG-Based Schizophrenia Detection: A Replication and Critical Evaluation of Gosala et al. (2023)*
> **Author:** Bamdad Booyeh · [GitHub](https://github.com/BamdadBooyeh)

---

## Overview

This repository fully reproduces the experiments described in the thesis. We replicate and critically evaluate **Gosala et al. (2023)** — *"Wavelet transforms for feature engineering in EEG data processing: An application on schizophrenia"* — which reported **97.98% classification accuracy** for automated schizophrenia detection from EEG.

### Three key findings

| Finding | Result |
| --- | --- |
| **Data leakage** | The original paper's 97.98% accuracy is inflated by up to **30.43 pp** via epoch-level cross-validation. Our leaky replication reached **98.13%**, while honest subject-independent validation drops to **63.52%–79.47%** for classical models. |
| **Feature complexity plateau** | CWT (2,603-dim), DWT (1,368-dim), WST (456-dim), and Raw Stats (228-dim) all converge around an honest ceiling of **~79%**. Wavelet engineering yielded a **-0.5 pp** uplift compared to a simple time-domain baseline. |
| **Neural networks** | **EEGNet** achieves **90.66% ± 7.48%** under honest validation (+11.19 pp over best classical method). However, catastrophic fold drops reveal structural fragility due to small subject sample sizes. |

---

## Repository Structure

```
eeg-schizophrenia-replication/
│
├── README.md
├── LICENSE
├── requirements.txt         ← pip install -r requirements.txt
├── environment.yml          ← conda env create -f environment.yml
│
├── notebooks/
│   └── schizophrenia_eeg_replication.ipynb   ← main notebook (all experiments)
│
├── figures/                        ← output figures (auto-generated)
│   └── .gitkeep
│
├── cache/                          ← extracted features cached here (auto-generated)
│   └── .gitkeep
│
└── data/
    └── README_DATA.md              ← dataset download instructions

```

---

## Dataset

**Warsaw EEG Dataset** — publicly available, no registration required.

| Property | Value |
| --- | --- |
| Subjects | 28 total (14 Healthy Controls, 14 Schizophrenia patients) |
| Channels | 19 (10-20 system) |
| Sampling rate | 250 Hz |
| Format | European Data Format (.edf) |
| Total epochs | 7,201 (5s windows, 1s overlap: 3,251 Healthy, 3,950 Schizophrenia) |

**Download:** [https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441)

After downloading, set `DATA_DIR` in Cell 3 of the notebook to the folder containing the 28 `.edf` files.

---

## Quickstart

### Option A — Google Colab (recommended, no setup required)

1. Open the notebook in Colab via the badge above.
2. Mount your Google Drive.
3. Upload the Warsaw EEG `.edf` files to Drive.
4. Edit `DATA_DIR` and `CACHE_DIR` in **Cell 3**.
5. Run cells top to bottom.

**First run:** ~30 min (feature extraction). All features cached automatically.

**Subsequent runs:** Skip Cell 5 → features reload in seconds from cache.

### Option B — Local (Python 3.12)

```bash
# Clone the repository
git clone https://github.com/BamdadBooyeh/eeg-schizophrenia-replication.git
cd eeg-schizophrenia-replication

# Install dependencies
pip install -r requirements.txt

# Edit paths in Cell 3 of the notebook, then run
jupyter notebook notebooks/schizophrenia_eeg_replication.ipynb

```

### Option C — Conda environment

```bash
conda env create -f environment.yml
conda activate eeg-schizo
jupyter notebook notebooks/schizophrenia_eeg_replication.ipynb

```

---

## Notebook Structure

Run cells **in order**. Each cell is self-contained with comments.

| Cell | Description | Time |
| --- | --- | --- |
| 1 | Install dependencies | ~2 min |
| 2 | Imports | Instant |
| 3 | Configuration — **edit paths here** | Instant |
| 4 | Feature extraction functions (CWT, DWT, WST, Raw) | Instant |
| 5 | Load EDF files + extract all features (**run once**) | ~30 min |
| 6 | Reload saved features (use instead of Cell 5) | ~5 sec |
| 7 | Subject-independent 10-fold CV builder | Instant |
| 8 | **Finding 1:** Data leakage demonstration | ~10 min |
| 9–10 | Classical ML — CWT, DWT, WST features | ~45 min |
| 11 | Full leakage gap matrix (heatmap) | ~20 min |
| 12 | Neural network infrastructure (shared classes) | Instant |
| 12a | MLP on CWT features | ~10 min |
| 12b | MLP on WST features | ~10 min |
| 12c | ShallowConvNet on raw EEG | ~15 min |
| 12d | EEGNet on raw EEG (**best model**) | ~25 min |
| 12e | EEG-TCNet on raw EEG | ~30 min |
| 13 | Aggregate NN results + save | Instant |
| 14 | Final visualisation dashboard | ~1 min |
| 15 | **Raw statistical baseline** (no wavelet) | ~15 min |

**Skip Cell 5 on return visits** — features are cached in `CACHE_DIR`.

---

## Results Summary

### Experiment 1: Data Leakage Demonstration

This experiment highlights the "hallucination of performance" caused by epoch-level splitting rather than maintaining subject-independent validation boundaries.

| Classifier | Leaky CV Accuracy (Epochs) | Honest CV Accuracy (Subjects) | Accuracy Gap |
| --- | --- | --- | --- |
| **Decision Tree + CWT** | 93.95% ± 0.68% | 63.52% ± 10.00% | -30.43 pp |
| **Random Forest + CWT** | 98.13% ± 0.68% | 77.97% ± 11.25% | -20.16 pp |
| **Paper Reported (DT + CWT)** | 97.98% | — | — |

> **Interpretation:** Our leaky Random Forest (98.13%) exceeded the original paper's reported Decision Tree (97.98%). This confirms that these near-perfect scores are artifacts of models memorizing individual "brain fingerprints" rather than identifying generalizable disease markers.

---

### Classical Machine Learning Results (Honest CV)

The following results were obtained under strict subject-independent validation (`GroupKFold`).

#### 1. Continuous Wavelet Transform (CWT) — 2,603 Features

| Classifier | Accuracy | Sensitivity | Specificity | Kappa (κ) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 71.84% ± 14.7% | 69.5% | 68.3% | 0.381 |
| SVM (RBF) | 73.23% ± 13.9% | 72.9% | 68.6% | 0.417 |
| Decision Tree | 63.52% ± 10.0% | 74.1% | 50.4% | 0.244 |
| **Random Forest ★** | **79.17% ± 10.8%** | **88.0%** | **67.1%** | **0.557** |
| AdaBoost | 72.62% ± 14.5% | 80.7% | 60.0% | 0.412 |
| XGBoost | 73.49% ± 14.0% | 82.7% | 59.0% | 0.423 |

#### 2. Discrete Wavelet Transform (DWT) — 1,368 Features

| Classifier | Accuracy | Sensitivity | Specificity | Kappa (κ) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 72.75% ± 14.9% | 71.5% | 69.7% | 0.413 |
| **SVM (RBF) ★** | **73.55% ± 15.0%** | **75.4%** | **67.8%** | **0.433** |
| Decision Tree | 64.43% ± 12.8% | 73.9% | 51.7% | 0.257 |
| Random Forest | 70.73% ± 12.2% | 78.9% | 58.2% | 0.376 |
| AdaBoost | 70.53% ± 15.0% | 78.2% | 57.7% | 0.364 |
| XGBoost | 71.62% ± 11.0% | 83.2% | 54.4% | 0.384 |

#### 3. Wavelet Scattering Transform (WST) — 456 Features

| Classifier | Accuracy | Sensitivity | Specificity | Kappa (κ) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 74.50% ± 15.2% | 72.9% | 72.3% | 0.455 |
| SVM (RBF) | 75.71% ± 13.5% | 78.3% | 69.0% | 0.478 |
| Decision Tree | 67.09% ± 13.0% | 77.1% | 53.9% | 0.313 |
| Random Forest | 75.14% ± 11.9% | 81.0% | 64.7% | 0.465 |
| **AdaBoost ★** | **77.37% ± 15.8%** | **83.5%** | **66.2%** | **0.503** |
| XGBoost | 76.04% ± 12.5% | 85.5% | 60.8% | 0.474 |

#### 4. Raw Statistical Baseline — 228 Features (No Wavelet Transform)

| Classifier | Accuracy | Sensitivity | Specificity | Kappa (κ) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 73.22% ± 15.1% | 76.8% | 65.4% | 0.424 |
| SVM (RBF) | 73.25% ± 12.7% | 76.8% | 66.0% | 0.431 |
| Decision Tree | 72.74% ± 9.4% | 85.3% | 57.8% | 0.431 |
| Random Forest | 79.25% ± 13.5% | 85.6% | 69.9% | 0.560 |
| AdaBoost | 77.48% ± 15.4% | 84.3% | 66.2% | 0.508 |
| **XGBoost ★** | **79.47% ± 13.5%** | **88.5%** | **66.1%** | **0.554** |

> **Interpretation (Feature Complexity Plateau):** All four extraction approaches converged at an honest ceiling of **~79%**. The calculated "Wavelet Uplift" (gain from wavelets over raw baseline statistics) stands at **-0.5 pp** (78.97% best wavelet vs 79.47% raw stats), proving that at $n=28$, thousands of complex spectral features act as redundant noise compared to basic time-domain statistics like MAD and standard deviation. Pairwise differences are statistically insignificant (Wilcoxon signed-rank $p > 0.05$).

---

### End-to-End Neural Network Results (Honest CV)

Evaluated with Focal Loss, AdamW, and Cosine Annealing.

| Architecture | Input Representation | Mean Accuracy | Sensitivity | Specificity | Kappa (κ) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **MLP** | CWT features | 81.05% ± 13.0% | 84.7% | 73.5% | 0.587 | 0.915 |
| **MLP** | WST features | 81.97% ± 13.4% | 81.0% | 80.4% | 0.615 | 0.893 |
| **ShallowConvNet** | Raw EEG | 86.34% ± 14.9% | 85.9% | 83.3% | 0.697 | 0.932 |
| **EEG-TCNet** | Raw EEG | 86.80% | — | — | — | 0.951 |
| **EEGNet ★** | **Raw EEG** | **90.66% ± 7.48%** | **87.9%** | **92.2%** | **0.805** | **0.960** |

> **Interpretation:** EEGNet achieved a genuine **+11.19 pp** improvement over the best classical method, proving that deep end-to-end learning architectures reliably isolate spatial-temporal signatures lost in manual feature engineering. However, an acute performance drop in Fold 6 (73.3% accuracy, 39.7% sensitivity) reveals the vulnerability of deep models when subject diversity is low.

---

## Feature Dimensions

| Method | Stage 1 | Stage 2 | Total Dimensions |
| --- | --- | --- | --- |
| **RAW** | 19 channels × 12 stats | — | **228** |
| **CWT** | 19 channels × 12 stats | 19 channels × 125 mean power (mexh) | **2,603** |
| **DWT** | 19 channels × 12 stats | 19 channels × 5 subbands × 12 stats (db4, L=4) | **1,368** |
| **WST** | 19 channels × 12 stats | 19 channels × 12 scattering stats (J=6, Q=8) | **456** |

---

## Neural Network Architectures

| Model | Parameters | Input Shape | Key Architectural Design |
| --- | --- | --- | --- |
| **FeatureMLP** | ~1.4M | `(B, feat_dim)` | 3-layer Fully Connected + BatchNorm |
| **ShallowConvNet** | ~45K | `(B, 19, 1250)` | Square + log band-power approximation |
| **EEGNet** | **2,033** | `(B, 19, 1250)` | Compact depthwise & separable convolutions |
| **EEG-TCNet** | ~8K | `(B, 19, 1250)` | EEGNet spatial core + Temporal Convolutional Network |

All neural networks are trained uniformly using: Focal Loss · AdamW · Cosine Annealing LR Schedule · Early Stopping · Weighted Random Sampler.

---

## Reproducibility Notes

* **Random Seeds:** `random_state=0` for all scikit-learn algorithms, and `random_state=42` for GroupKFold validation splitting.
* **Feature Cache:** Running Cell 5 saves `features.npz` and `X_raw_epochs.npy` to local storage, producing identical arrays on any system.
* **CV Folds:** Deterministic round-robin subject assignment guarantees reproducible cross-validation fold subsets.
* **Hardware Acceleration:** All models detect and run automatically on execution targets (`torch.cuda.is_available()`).
* **Assertion Validations:** Script strictly expects exactly **7,201** total epochs (3,251 Healthy Controls, 3,950 Schizophrenia subjects) checked via assertions in Cell 5.

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{booyeh2025,
  author    = {Booyeh, Bamdad},
  title     = {Feature Complexity vs Sample Size in EEG-Based
               Schizophrenia Detection: A Replication and Critical
               Evaluation of Gosala et al. (2023)},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/BamdadBooyeh/eeg-schizophrenia-replication}
}

```

Original paper being replicated:

```bibtex
@article{gosala2023,
  author  = {Gosala, Bhargav and Bhatt, Jaymin and Bhatt, Ankit
             and Sharma, Prashant and Shukla, Alok},
  title   = {Wavelet transforms for feature engineering in {EEG}
             data processing: An application on schizophrenia},
  journal = {Biomedical Signal Processing and Control},
  volume  = {85},
  pages   = {104811},
  year    = {2023},
  doi     = {10.1016/j.bspc.2023.104811}
}

```

---

## License

MIT License — see [LICENSE](https://www.google.com/search?q=LICENSE)
