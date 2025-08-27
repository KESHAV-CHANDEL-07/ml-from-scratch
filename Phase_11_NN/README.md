# 🧠 AI-Driven Personalized Skincare Diagnosis and Recommendation System

## 🧭 Objective

This project aims to build a **clinically trustworthy, fairness-aware, and privacy-respecting ML system** that:
- Diagnoses facial skin conditions (e.g., acne, pigmentation, dryness)
- Incorporates lifestyle data (e.g., sleep, skincare habits, SPF use)
- Recommends only **necessary** skincare treatments or products
- Ensures performance equity across **diverse skin tones**

---

## 🔧 Technical Architecture

### 🔹 Multimodal Input
- **Facial Images**: Multi-angle facial images captured in natural lighting
- **Lifestyle Data**: Structured inputs (sleep, sunscreen usage, diet, skin sensitivity, etc.)

### 🔹 Modeling Strategy

| Component        | Model / Technique                                           |
|------------------|-------------------------------------------------------------|
| 🖼️ Image Encoder | `Swin Transformer` or `EfficientNetV2` + (optional) `DINOv2` |
| 📊 Tabular Encoder | `TabTransformer` or `CatBoost`                             |
| 🔗 Fusion Layer   | Concatenation + `Cross-Attention` / `Residual Dense Blocks` |
| 🎯 Output Heads   | Multi-task: skin condition detection, severity regression, and product recommendation |

---

## 🧪 Training Strategy

### Phase 1: Self-Supervised Pretraining
- Pretrain image encoder with **DINOv2** or **SimCLR** on unlabeled skin images

### Phase 2: Supervised Fine-tuning
- Train on public medical datasets:
  - `HAM10000`
  - `ISIC Archive`
  - Custom-labeled face datasets (via dermatologist collaboration)

---

## ⚖️ Fairness & Bias Mitigation

- Evaluate and debias across **Monk Skin Tone Scale** and **Fitzpatrick Skin Types**
- Use:
  - `GroupDRO` for worst-group error minimization
  - `Adversarial Debiasing` to avoid skin tone shortcut learning
- Model evaluated for **parity across ethnic groups**

---

## 🧴 Personalized Recommendation Engine

- **Hybrid recommender**:
  - Rule-based filtering (e.g. acne → no comedogenic oils)
  - Reinforcement Learning (`Deep Q-Learning`, `Bandits`) for optimal long-term skincare strategy
- Personalized recommendations improve with **user feedback** (ratings, skin improvement logs)

---

## 🔍 Explainability

- Visual: `Grad-CAM++`, `Saliency Maps`
- Tabular: `SHAP`, `LIME`
- Output includes:
  - Region-based diagnosis heatmaps
  - Confidence scores for transparency

---

## 🔒 Privacy & Clinical Trust

- Fully on-device or encrypted cloud processing
- No facial image is stored or reused without user consent
- System includes **"no recommendation needed"** output to prevent overtreatment
- Designed with input from **dermatologists and skin experts**

---

## 🛠️ Tech Stack

| Task                        | Tool / Framework              |
|-----------------------------|-------------------------------|
| Vision Modeling             | `PyTorch`, `Swin-T`, `DINOv2` |
| Tabular Modeling            | `TabTransformer`, `CatBoost` |
| RL Recommender              | `Stable-Baselines3`, `Ray RLlib` |
| Explainability              | `SHAP`, `Grad-CAM`, `Captum`  |
| Deployment                  | `FastAPI`, `ONNX`, `Streamlit` |
| Tracking                    | `Weights & Biases`            |

---

## 📅 Roadmap

- ✅ **Phase 1**: Architecture setup, dataset collection, baseline training
- 🔜 **Phase 2**: Dermatologist feedback, fairness validation
- 🔜 **Phase 3**: Reinforcement learning recommender + app/web interface

---

## 🤝 Collaboration

Interested in contributing or collaborating (especially dermatologists or med students)?  
## Email:  keshavchandel05@gmail.com

