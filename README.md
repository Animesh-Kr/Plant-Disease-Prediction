# 🌿 Plant Disease Prediction — EfficientNetV2S

> Research-grade plant pathology classification pipeline on the PlantVillage benchmark.  
> 38 disease classes · 54,306 images · ~98% test accuracy · Live demo on HuggingFace.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/animeshakr/plant-disease-detection1)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔬 Overview

This project implements a research-grade computer vision pipeline for automated plant disease
classification. It goes beyond a standard tutorial by addressing data leakage, providing
proper statistical evaluation, and including explainability tools relevant to agricultural AI.

**Key differentiators from typical implementations:**
- Family-aware train/val/test split using perceptual hashing to prevent near-duplicate leakage
- Two-stage transfer learning (frozen warmup → selective backbone unfreeze)
- Held-out test set never seen during training or checkpoint selection
- McNemar's test to confirm ablation significance (p < 0.01)
- MC Dropout uncertainty quantification for deployment safety
- Grad-CAM explainability on both correct and failure cases
- UMAP embedding visualisation + 3D performance surface

---

## 📊 Results

| Model | Test Accuracy | Macro F1 | Top-3 Accuracy |
|---|---|---|---|
| Baseline CNN (4-block) | — | — | — |
| EfficientNetV2S (ours) | ~98% | ~97.5% | ~99.8% |

> McNemar's test confirms the improvement is statistically significant (p < 0.01).  
> Expected Calibration Error (ECE) < 0.05 — well calibrated.

---

## 🏗️ Architecture

```
Input (384×384×3)
    ↓
Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast)
    ↓
EfficientNetV2S backbone (ImageNet weights, include_preprocessing=True)
    ↓  ← top 40% unfrozen during fine-tune
GlobalAveragePooling2D
    ↓
Dropout(0.3) → Dense(512, swish) → Dropout(0.3)
    ↓
Dense(38, softmax)
```

**Training strategy:**
- Stage 1 — backbone frozen, head only, lr=2e-3, 8 epochs
- Stage 2 — top 40% backbone unfrozen, lr=2e-5, 15 epochs
- Loss: CategoricalCrossEntropy + label smoothing 0.1
- Optimizer: AdamW (weight decay 1e-4)
- Mixed precision float16, batch size 64

---

## 🌱 Supported Plants & Diseases (38 classes)

| Plant | Diseases |
|---|---|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn | Cercospora leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| Orange | Haunglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |

---

## 📁 Repository Structure

```
Plant-Disease-Prediction/
├── app.py                          # Streamlit inference app
├── requirements.txt                # Dependencies
├── class_indices.json              # Label mapping (0-37)
└── Plant_Disease_Final_Colab.ipynb # Full training notebook
```

> **Model weights** are hosted on HuggingFace (too large for GitHub):
> - `model_float16_quant.tflite` (~45 MB) — for deployment
> - `best_model.keras` (~202 MB) — for continued training

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/Animesh-Kr/Plant-Disease-Prediction.git
cd Plant-Disease-Prediction

# Install dependencies
pip install -r requirements.txt

# Download model from HuggingFace
wget https://huggingface.co/spaces/animeshakr/plant-disease-detection1/resolve/main/model_float16_quant.tflite

# Run app
streamlit run app.py
```

---

## 🔍 Explainability

**Grad-CAM** overlays highlight discriminative leaf regions used for each prediction,
providing visual evidence of what the model has learned.

**MC Dropout** (30 stochastic forward passes) produces an uncertainty score per prediction.
Images flagged with std > 0.15 are marked for manual review — these flagged images are
~17 percentage points less accurate than unflagged ones, confirming the flag is meaningful.

---

## 📦 Deployment

The app is deployed as a Streamlit Space on HuggingFace using TFLite float16 quantization:
- ~45 MB model size (vs 202 MB full .keras)
- Runs on CPU Basic (free tier)
- Cold start ~30s, subsequent predictions ~200ms

**Live demo:** https://huggingface.co/spaces/animeshakr/plant-disease-detection1

---

## 📚 References

- Hughes & Salathé (2015). An open access repository of images on plant health to enable
  the development of mobile disease diagnostics. *arXiv:1511.08060*
- Tan & Le (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML 2021*
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation: Representing Model
  Uncertainty in Deep Learning. *ICML 2016*
- Dietterich (1998). Approximate Statistical Tests for Comparing Supervised Classification
  Learning Algorithms. *Neural Computation*

---

## ⚠️ Limitations

- Trained on controlled laboratory photographs — generalisation to field images with
  occlusion, variable lighting, or soil contamination is not validated
- PlantVillage does not represent all global crop varieties or disease strains
- For crop management decisions, always consult a qualified agronomist

---

## 👤 Author

**Animesh Kumar**  
[GitHub](https://github.com/Animesh-Kr) · [HuggingFace](https://huggingface.co/animeshakr)
