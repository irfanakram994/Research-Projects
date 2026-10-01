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
# 3. 📈 BondMM Interactive Simulator & Analytics Dashboard

An interactive simulation and comparative analytics platform built for **Google Colab** using **Streamlit**, **Plotly**, and **Prophet**. This project models, visualizes, and benchmark tests the Automated Market Maker (AMM) protocol introduced in the research paper:

> **"An Automated Market Maker Algorithm for Fixed-Rate Trading with Flexible Maturities"**  
> *Authors: Tuan Tran and Duc A. Tran*

---

## 📌 Project Overview

Traditional fixed-rate DeFi protocols (e.g., Yield Protocol, Notional Finance) rely on discrete maturity buckets, leading to fragmented liquidity pools and severe capital inefficiency. **BondMM** addresses these challenges by introducing a continuous yield-space invariant function $Kx^\alpha + y^\alpha = C$, enabling:

1. **Flexible, Continuous Maturities:** Trade bonds across any maturity duration without fragmented liquidity pools.
2. **Single-Transaction Cross-Maturity Swaps:** Shift bond maturity positions from $T_1$ to $T_2$ seamlessly without multi-pool routing.
3. **LP Equity Capital Protection:** Built-in dynamic equity ratio checks ($\rho \ge 90\%$) that protect liquidity providers from impermanent loss and systemic drawdown during extreme interest rate volatility.
4. **Predictive Rate Forecasting:** Integrates Meta's **Prophet** time-series forecasting model alongside traditional **Vasicek stochastic models** to simulate future interest rate trajectories (e.g., AAVE borrow rates) and stress-test the AMM invariant curves.

---

## 🚀 Key Features

* **Interactive Order Sandbox:** Place live `Borrow`, `Lend`, and `Cross-Maturity Swap` orders with real-time calculations for price impact, implied fixed rates, and virtual state updates.
* **2D Invariant & Yield Curve Plotter:** Dynamic Plotly visualizations showing real-time state movements along the $Kx^\alpha + y^\alpha = C$ invariant curve and corresponding yield curve updates.
* **Prophet Rate Engine:** Time-series interest rate forecasting engine to project yield trends and uncertainty intervals ($y_{hat}$, $y_{hat\_lower}$, $y_{hat\_upper}$).
* **Comparative Benchmark Suite:** Head-to-head performance comparison of **BondMM vs. Yield Protocol vs. Notional Finance** measuring execution slippage and LP equity resilience under rate shocks.

---

## 🛠️ System Architecture
┌                   ──────────────────────────────────────┐
                   │        Google Colab Runtime          │
                   └──────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
   ┌──────────▼──────────┐                         ┌──────────▼──────────┐
   │     BondMM Core     │                         │     Rate Engine     │
   │  - Invariant Curve  │                         │  - Vasicek Model    │
   │  - Virtual States   │                         │  - Prophet Forecast │
   │  - Cross Swaps      │                         │  - Synthetic Data   │
   └──────────┬──────────┘                         └──────────┬──────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                           ┌──────────▼──────────┐
                           │       app.py        │
                           │ (Streamlit Interface)│
                           └──────────┬──────────┘
                                      │
                           ┌──────────▼──────────┐
                           │    localtunnel      │
                           │  (Public Web URL)   │
                           └─────────────────────┘
---

## 💻 How to Run on Google Colab

This project is optimized to run entirely inside **Google Colab (Free Tier)** using in-memory execution—no Google Drive storage or local setup required.

### Step 1: Open Google Colab & Install Dependencies
Open a fresh notebook in Google Colab and run the following in the first cell:

```python
!pip install -q streamlit pyngrok numpy pandas plotly scipy prophet
```
import subprocess

# Launch Streamlit server in the background
p = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

# Expose local Streamlit port via localtunnel
!npx localtunnel --port 8501

## Author

Irfan Ali
