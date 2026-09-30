# 1. 3D Human Pose Estimation using YOLO11

This project performs human pose estimation using YOLO11 Pose.

## Features

- Human keypoint detection
- Pose visualization
- Joint angle calculation
- Video processing
- Exercise analysis

## Technologies Used

- Python
- YOLO11 Pose
- OpenCV
- NumPy
- Matplotlib
- Google Colab

## Colab Notebook

https://colab.research.google.com/drive/1HJPgNdoDxoU0N9UTMFJRK_H9yJfWrEx-?usp=sharing

## Sample Outputs

Project outputs are available here:

https://drive.google.com/drive/folders/1rMzeOmaq52YxXo6190A5G5XI5aSGgJWC?usp=sharing

## Future Work

- 3D pose estimation
- Exercise classification
- Real-time fitness coach
- Streamlit dashboard

# 2. SkillNet Proof-of-Concept Baseline

A lightweight PyTorch proof-of-concept inspired by the paper **"SkillNet: Open-Style Skill Acquisition and Adaptive Inference for Robust Biomedical Deep Learning"** (BCB '26). 

This project explores the core principles of sequential hard-sample training and confidence-guided adaptive fallback routing on biomedical image classification benchmarks.

---

## 📌 Overview

Standard deep learning backbones often struggle with catastrophic forgetting when adapted sequentially or lose confidence on ambiguous, out-of-distribution test samples. 

This repository implements a simplified two-stage pipeline:
1. **Base Model 1:** Fine-tuned on the primary dataset (`BloodMNIST`).
2. **Base Model 2:** Fine-tuned exclusively on training samples misclassified by Base Model 1 (Hard-Sample Mining).
3. **Adaptive Inference Engine:** Routes test inputs based on a confidence threshold ($\alpha = 0.70$). Inputs where Base Model 1's softmax confidence falls below $\alpha$ are routed to Base Model 2.

---

## 📊 Preliminary Results (BloodMNIST)

Evaluated on the full `BloodMNIST` test set (**3,421 samples**):

| Model Setup | Test Accuracy | Notes |
| :--- | :--- | :--- |
| **Base Model 1** | **92.81%** | Primary fine-tuned backbone |
| **Base Model 2** | **15.93%** | Standalone accuracy (Trained *only* on hard samples) |
| **SkillNet Adaptive Engine ($\alpha=0.70$)** | **90.50%** | Fallback Trigger Rate / Miss Rate: **9.53%** (326/3,421) |

---

## 🔍 Key Insights & Analysis

1. **Hard-Sample Specialization vs. Overfitting:**  
   Base Model 2 achieves low standalone accuracy (15.93%) because it was trained exclusively on Model 1's hard cases without mixing general class distributions. This extreme specialization causes catastrophic forgetting on easy/general test cases.

2. **Motivation for SkillNet Mechanisms:**  
   Because Model 2 produces occasional false positives when fallback is triggered, routing low-confidence samples to a standalone secondary model slightly impacts overall accuracy. 
   
   This empirical finding demonstrates **why SkillNet's core contributions are essential**:
   * **Algorithm 1 (Layer-wise Skill Pruning & Merging):** Instead of maintaining independent models, distinct weight modifications must be pruned and merged into a unified master graph.
   * **Algorithm 3 (Dynamic Logit Masking):** Inference requires candidate logit selection rather than direct model switching to prevent false-positive fallbacks.

---

## 🛠️ Project Structure & Setup

This notebook is designed to run entirely in memory within **Google Colab (Free T4 GPU)** with **zero persistent local disk or Google Drive footprint**.

### Dependencies
- `torch` & `torchvision`
- `timm`
- `medmnist`
- `matplotlib`

```bash
pip install medmnist timm torch torchvision matplotlib

```
## Author

Irfan Ali
