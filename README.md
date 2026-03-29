---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🌿 Plant Disease Detection

Automated plant disease classification using **EfficientNetV2S** fine-tuned on the
[PlantVillage benchmark](https://arxiv.org/abs/1511.08060).

## Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNetV2S |
| Input resolution | 384 × 384 |
| Classes | 38 plant diseases |
| Dataset | PlantVillage (54,306 images) |
| Export format | TFLite float16 |
| Test accuracy | ~98% |

## Supported Plants & Diseases

Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach ·
Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato

## Training Pipeline

- Family-aware 70/15/15 split with perceptual-hash near-duplicate removal
- Two-stage transfer learning: frozen warmup → top-40% backbone fine-tune
- Label smoothing 0.1, AdamW, mixed precision float16
- Evaluated on held-out test set only (zero validation leakage)

## References

- Hughes & Salathé (2015). An open access repository of images on plant health. arXiv:1511.08060
- Tan & Le (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021

## Limitations

Trained on controlled laboratory images. Performance on field photographs
with occlusion, soil contamination, or variable lighting is not validated.
