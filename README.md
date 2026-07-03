# EEG-Based Schizophrenia Detection: Replication & Critical Evaluation

> **Thesis:** *Feature Complexity vs Sample Size in EEG-Based Schizophrenia Detection: A Replication and Critical Evaluation of Gosala et al. (2023)*
>
> **Author:** Bamdad Booyeh · [GitHub](https://github.com/BamdadBooyeh)
> **Institution:** Sapienza University of Rome

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Overview

This repository fully reproduces the experiments described in the thesis. We replicate and critically evaluate **Gosala et al. (2023)** — *"Wavelet transforms for feature engineering in EEG data processing: An application on schizophrenia"* — which reported **97.98% classification accuracy** for automated schizophrenia detection from EEG.

### Three key findings

| Finding | Result |
|---|---|
| **Data leakage** | Paper's 97.98% inflated by ~30 pp via epoch-level CV. Our leaky replication: 98.13%. Honest subject-independent CV: 63–79%. |
| **Feature complexity plateau** | CWT (2,603-dim), DWT (1,368-dim), WST (456-dim) and raw stats (228-dim) all achieve statistically indistinguishable accuracy at n=28 (Wilcoxon p > 0.05). |
| **Neural networks** | EEGNet achieves **90.66% ± 7.48%** under honest CV — +11.49 pp over best classical method — but high fold variance confirms sample size remains the binding constraint. |

---

## Repository Structure

```
eeg-schizophrenia-replication/
│
├── README.md
├── LICENSE
├── requirements.txt               ← pip install -r requirements.txt
├── environment.yml                ← conda env create -f environment.yml
│
├── notebooks/
│   └── schizophrenia_eeg_replication.ipynb   ← main notebook (all experiments)
│
├── figures/                       ← output figures (auto-generated)
│   └── .gitkeep
│
├── cache/                         ← extracted features cached here (auto-generated)
│   └── .gitkeep
│
└── data/
    └── README_DATA.md             ← dataset download instructions
```

---

## Dataset

**Warsaw EEG Dataset** — publicly available, no registration required.

| Property | Value |
|---|---|
| Subjects | 28 (14 healthy, 14 schizophrenia) |
| Channels | 19 (10-20 system) |
| Sampling rate | 250 Hz |
| Format | European Data Format (.edf) |
| Total epochs | 7,201 (5s, 1s overlap) |

**Download:** https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441

After downloading, set `DATA_DIR` in Cell 3 of the notebook to the folder containing the 28 `.edf` files.

---

## Quickstart

### Option A — Google Colab (recommended, no setup required)

1. Open the notebook in Colab via the badge above
2. Mount your Google Drive
3. Upload the Warsaw EEG `.edf` files to Drive
4. Edit `DATA_DIR` and `CACHE_DIR` in **Cell 3**
5. Run cells top to bottom

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
|---|---|---|
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

### Classical ML — Honest Subject-Independent CV

| Method | Features | Best Classifier | Accuracy | Kappa |
|---|---|---|---|---|
| RAW (no wavelet) | 228 | Random Forest | ~75% | ~0.49 |
| CWT | 2,603 | Random Forest | **79.17%** | 0.557 |
| DWT | 1,368 | AdaBoost | ~77% | ~0.50 |
| WST | 456 | AdaBoost | 77.37% | 0.503 |

> All pairwise differences: Wilcoxon signed-rank p > 0.05 (not significant)

### Neural Networks — Honest Subject-Independent CV

| Architecture | Input | Accuracy | AUC | Kappa |
|---|---|---|---|---|
| MLP | CWT features | 81.05% ± 13.0% | 0.915 | 0.587 |
| MLP | WST features | 81.97% ± 13.4% | 0.893 | 0.615 |
| ShallowConvNet | Raw EEG | 86.34% ± 14.9% | 0.932 | 0.697 |
| **EEGNet** | **Raw EEG** | **90.66% ± 7.48%** | **0.960** | **0.805** |
| EEG-TCNet | Raw EEG | — | — | — |

### Data Leakage

| CV Protocol | Decision Tree + CWT | Random Forest + CWT |
|---|---|---|
| Leaky (epoch-level, paper-style) | 93.95% | **98.13%** |
| Honest (subject-independent) | 63.52% | 79.17% |
| **Gap** | **~30 pp** | **~19 pp** |
| Paper reported | 97.98% | — |

---

## Feature Dimensions

| Method | Stage 1 | Stage 2 | Total |
|---|---|---|---|
| RAW | 19×12 stats | — | **228** |
| CWT | 19×12 stats | 19×125 mean power (mexh) | **2,603** |
| DWT | 19×12 stats | 19×5×12 subband stats (db4, L=4) | **1,368** |
| WST | 19×12 stats | 19×12 scattering stats (J=6, Q=8) | **456** |

---

## Neural Network Architectures

| Model | Params | Input | Key design |
|---|---|---|---|
| FeatureMLP | ~1.4M | (B, feat_dim) | 3-layer FC + BatchNorm |
| ShallowConvNet | ~45K | (B, 19, 1250) | Square+log band-power approx |
| EEGNet | **2,033** | (B, 19, 1250) | Depthwise separable conv |
| EEG-TCNet | ~8K | (B, 19, 1250) | EEGNet spatial + TCN temporal |

All models use: Focal loss · AdamW · Cosine annealing LR · Early stopping · Weighted sampler

---

## Reproducibility Notes

- **Random seeds:** `random_state=0` for all sklearn models, `random_state=42` for fold splits
- **Feature cache:** Cell 5 saves `features.npz` and `X_raw_epochs.npy` — identical on any machine with the same dataset
- **CV folds:** deterministic round-robin subject assignment — same fold composition every run
- **GPU/CPU:** All models run on CPU or GPU automatically (`torch.cuda.is_available()`)
- **Expected total epochs:** 7,201 (H=3,251, S=3,950) — assertion in Cell 5 confirms this

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

MIT License — see [LICENSE](LICENSE)
