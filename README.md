# EEG-Based Schizophrenia Detection: Replication & Critical Evaluation

> **Thesis:** Wavelet-Based EEG Classification for Schizophrenia Detection **
>
> **Author:** Bamdad Booyeh · [GitHub](https://github.com/BamdadBooyeh)


[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BamdadBooyeh/EEG-Classification-for-Schizophrenia/blob/main/schizophrenia_eeg_detection.ipynb)

---

## Overview

This repository fully reproduces the experiments from the thesis.
We replicate and critically evaluate **Gosala et al. (2023)** —
*"Wavelet transforms for feature engineering in EEG data processing:
An application on schizophrenia"* — which reported **97.98% accuracy**
for automated schizophrenia detection from EEG.

### Three key findings

| # | Finding | Result |
|---|---|---|
| 1 | **Data leakage** | Paper's 97.98% inflated by ~30 pp via epoch-level CV. Our leaky replication: 98.13%. Honest subject-independent CV: 63–79%. |
| 2 | **Feature complexity plateau** | CWT (2,603-dim), DWT (1,368-dim), WST (456-dim) and raw stats (228-dim) all achieve statistically indistinguishable accuracy at n=28. Wilcoxon p > 0.05 for all pairwise comparisons. |
| 3 | **Neural networks** | EEGNet achieves **90.66% ± 7.48%** under honest CV — +11.49 pp above best classical method — but high fold variance confirms sample size is the binding constraint. |

---

## Repository Structure

```
EEG-Classification-for-Schizophrenia/
│
├── README.md
├── LICENSE
├── requirements.txt                        ← pip install -r requirements.txt
├── environment.yml                         ← conda env create -f environment.yml
│
├── schizophrenia_eeg_detection.ipynb       ← main notebook (all experiments)
│
├── figures/                                ← output figures (auto-generated)
│   └── .gitkeep
│
├── cache/                                  ← extracted features cached here
│   └── .gitkeep
│
└── data/
    └── README_DATA.md                      ← dataset download instructions
```

---

## Dataset

**Warsaw EEG Dataset** — publicly available, no registration required.

| Property | Value |
|---|---|
| Subjects | 28 (14 healthy, 14 schizophrenia) |
| Channels | 19 (10-20 system) |
| Sampling rate | 250 Hz |
| Recording | Resting state, eyes closed |
| Format | European Data Format (.edf) |
| Total epochs extracted | 7,201 (5 s, 1 s overlap) |

**Download:** https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441

See [`data/README_DATA.md`](data/README_DATA.md) for step-by-step download instructions.

---

## Quickstart

### Option A — Google Colab (recommended, no local setup needed)

Click the **Open in Colab** badge above, or open the link directly:

```
https://colab.research.google.com/github/BamdadBooyeh/EEG-Classification-for-Schizophrenia/blob/main/schizophrenia_eeg_detection.ipynb
```

Then:

1. Run **Cell 1** to install dependencies
2. **Cell 3** — mount Google Drive and edit `DATA_DIR` / `CACHE_DIR`
3. Upload the 28 Warsaw `.edf` files to the `DATA_DIR` folder on Drive
4. **Cell 5** — extract features (~30 min, runs once only)
5. On future sessions: skip Cell 5, run **Cell 6** to reload from cache

### Option B — Local (Python 3.12)

```bash
# 1. Clone
git clone https://github.com/BamdadBooyeh/EEG-Classification-for-Schizophrenia.git
cd EEG-Classification-for-Schizophrenia

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook schizophrenia_eeg_detection.ipynb
```

Edit `DATA_DIR` and `CACHE_DIR` in **Cell 3** to point to your local paths,
then run cells in order.

### Option C — Conda

```bash
conda env create -f environment.yml
conda activate eeg-schizo
jupyter notebook schizophrenia_eeg_detection.ipynb
```

---

## Notebook Structure

Run cells **in order top to bottom**. Each cell is self-contained with comments.

| Cell | Content | Est. time (GPU) |
|---|---|---|
| 1 | Install dependencies | ~2 min |
| 2 | Imports | Instant |
| **3** | **Configuration — edit `DATA_DIR` here** | Instant |
| 4 | Feature extraction functions (CWT, DWT, WST, Raw) | Instant |
| **5** | **Load EDF + extract features (run once only)** | ~30 min |
| **6** | **Reload saved features (use on return sessions)** | ~5 sec |
| 7 | Subject-independent 10-fold CV builder | Instant |
| 8 | Finding 1: Data leakage demonstration | ~10 min |
| 9–10 | Classical ML — CWT, DWT, WST features | ~45 min |
| 11 | Full leakage gap matrix heatmap | ~20 min |
| 12 | Neural network infrastructure (shared classes) | Instant |
| 12a | MLP on CWT features | ~10 min |
| 12b | MLP on WST features | ~10 min |
| 12c | ShallowConvNet on raw EEG | ~15 min |
| **12d** | **EEGNet on raw EEG (best model, 90.66%)** | ~25 min |
| 12e | EEG-TCNet on raw EEG | ~30 min |
| 13 | Aggregate NN results + save | Instant |
| 14 | Final visualisation dashboard | ~1 min |
| 15 | Raw statistical baseline (no wavelet) | ~15 min |

> **Tip:** On return sessions run Cell 2 → Cell 3 → Cell 6 → Cell 7, then jump
> directly to whichever experiment you need. All results are saved automatically.

---

## Results

### Classical ML — Honest 10-Fold Subject-Independent CV

| Method | Features | Best Classifier | Accuracy | Kappa |
|---|---|---|---|---|
| RAW (no wavelet) | 228 | Random Forest | ~75% | ~0.49 |
| DWT | 1,368 | AdaBoost | ~77% | ~0.50 |
| WST | 456 | AdaBoost | 77.37% | 0.503 |
| **CWT** | **2,603** | **Random Forest** | **79.17%** | **0.557** |

> All pairwise Wilcoxon signed-rank tests: p > 0.05 (no significant difference)

### Neural Networks — Honest 10-Fold Subject-Independent CV

| Architecture | Input | Accuracy | AUC | Kappa |
|---|---|---|---|---|
| MLP | CWT features | 81.05% ± 13.0% | 0.915 | 0.587 |
| MLP | WST features | 81.97% ± 13.4% | 0.893 | 0.615 |
| ShallowConvNet | Raw EEG | 86.34% ± 14.9% | 0.932 | 0.697 |
| **EEGNet** | **Raw EEG** | **90.66% ± 7.48%** | **0.960** | **0.805** |

### Data Leakage Analysis

| CV Protocol | Decision Tree + CWT | Random Forest + CWT |
|---|---|---|
| Leaky — epoch-level (paper-style) | 93.95% | 98.13% |
| Honest — subject-independent | 63.52% | 79.17% |
| **Leakage gap** | **~30 pp** | **~19 pp** |
| Paper reports | **97.98%** | — |

---

## Feature Dimensions

| Method | Stage 1 | Stage 2 | Total |
|---|---|---|---|
| RAW | 19 ch × 12 stats | — | **228** |
| CWT | 19 ch × 12 stats | 19 ch × 125 scales (mexh) | **2,603** |
| DWT | 19 ch × 12 stats | 19 ch × 5 subbands × 12 stats (db4, L=4) | **1,368** |
| WST | 19 ch × 12 stats | 19 ch × 12 scattering stats (J=6, Q=8) | **456** |

---

## Neural Network Architectures

| Model | Parameters | Input shape | Key design choice |
|---|---|---|---|
| FeatureMLP | ~1.4M | (B, feat\_dim) | 3-layer FC + BatchNorm + ELU |
| ShallowConvNet | ~45K | (B, 19, 1250) | Square + log band-power approximation |
| **EEGNet** | **2,033** | **(B, 19, 1250)** | **Depthwise separable convolution** |
| EEG-TCNet | ~8K | (B, 19, 1250) | EEGNet spatial + TCN dilated temporal |

All models trained with: Focal loss · AdamW · Cosine annealing LR ·
Early stopping · Weighted random sampler · Gradient clipping

---

## Reproducibility

| Factor | Value |
|---|---|
| Python | 3.12.13 |
| Random seed (sklearn) | `random_state=0` for all classifiers |
| Random seed (CV folds) | Round-robin, deterministic (no random state needed) |
| Feature cache | `features.npz` and `X_raw_epochs.npy` — identical on any machine |
| Expected total epochs | 7,201 (assertion in Cell 5 confirms this) |
| GPU / CPU | Auto-detected (`torch.cuda.is_available()`) |
| Operating system | Tested on Ubuntu 22.04 (Colab) and macOS 14 |

### Cached files (auto-generated, not in repo)

| File | Size | Description |
|---|---|---|
| `cache/features.npz` | ~500 MB | CWT + DWT + WST features for all 28 subjects |
| `cache/X_raw_epochs.npy` | ~1.7 GB | Raw EEG epochs (19 ch × 1250 samples) |
| `cache/classical_results.pkl` | ~5 MB | Classical ML fold results |
| `cache/nn_results.joblib` | ~10 MB | Neural network fold results |

These files are generated by Cell 5 and reloaded by Cell 6.
They are excluded from the repo via `.gitignore` due to size.

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
  url       = {https://github.com/BamdadBooyeh/EEG-Classification-for-Schizophrenia}
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
