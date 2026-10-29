# 🌾 AgroPulse AI: Multi-Modal Precision Agriculture & Agronomy Diagnostics Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework and diagnostic suite designed for **SIT-N Hack-O-Week (Weeks 17 & 18)**. 

AgroPulse AI addresses real-life precision farming challenges—combining leaf pathology computer vision, IoT microclimate time-series modeling, deep regularization ablation, transformer NLP triage, and LLM-driven multi-modal agronomic advisory reports.

---

## 📌 Syllabus & Implementation Matrix

| Requirement | Implementation Module | File / Script | Key Concepts Demonstrated |
| :--- | :--- | :--- | :--- |
| **1. Intro to PyTorch** | Soil & Photosynthesis Curve Optimization | `01_tensors_and_autograd/` | Tensors, GPU/CPU devices, autograd, forward/backward pass, custom loss, training loop |
| **2. CNNs** | AgriLeafNet Leaf Pathology Classifier | `02_cnn_classification/` | 2D Convolutions, Max Pooling, Strides, Receptive Field, Feature Maps, Softmax |
| **3. RNNs / LSTMs / GRUs** | Microclimate & Soil Telemetry Forecasting | `03_sequential_rnn_lstm_gru/` | Hidden states, vanishing gradients, gating mechanisms (LSTM vs GRU vs Vanilla RNN) |
| **4. Regularization in DL** | Empirical Regularization Laboratory | `04_regularization_techniques/` | Dropout, Batch Normalization, Early Stopping, Generalization gap visualization |
| **5. Transfer Learning** | Pretrained MobileNetV2 / ResNet18 | `05_transfer_learning_cnn/` | Frozen backbone feature extraction vs fine-tuning, adaptive pooling, classification head |
| **6. Hugging Face Transformers** | Agronomy Advisory & Symptom Triage | `06_transformers_nlp/` | Tokenization, attention mechanisms, sequence classification, zero-shot/pipeline inference |
| **7. LLM APIs & Integration** | Multi-Modal Agronomy Copilot Report Agent | `07_llm_agronomy_copilot/` | OpenAI/Anthropic/Groq/Ollama APIs, prompt engineering, structured synthesis, fallback engine |
| **8. Interactive Platform** | Full-Stack Streamlit & Web GUI | `08_interactive_dashboard/` | Real-time interactive playground for all 7 deep learning modules |

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run All Verification Tests
Ensure all 7 deep learning modules are verified:
```bash
python tests/test_all_modules.py
```

### 3. Launch the Interactive Dashboard
Launch the unified multi-module interactive GUI:
```bash
streamlit run 08_interactive_dashboard/app.py
```

---

## 📂 Project Architecture

```
Week_17_18_/
├── 01_tensors_and_autograd/       # Module 1: PyTorch Tensors, Autograd & Core Optimization
│   ├── tensors_autograd_demo.py   # Hands-on tensor creation, ops, and autograd
│   └── crop_yield_optimizer.py    # Non-linear crop yield model trained via custom loop
├── 02_cnn_classification/         # Module 2: CNN from Scratch for Image Classification
│   ├── dataset_generator.py       # Synthetic and augmentable leaf disease dataset generator
│   ├── cnn_model.py               # AgriLeafNet (Conv2D, BatchNorm, MaxPool, Linear)
│   └── train_cnn.py               # Training loop with accuracy, confusion matrix, feature maps
├── 03_sequential_rnn_lstm_gru/    # Module 3: Sequence Modeling for IoT Telemetry
│   ├── telemetry_dataset.py       # Microclimate & soil moisture sequence generator
│   ├── models.py                  # Vanilla RNN, LSTM, and GRU architectures
│   └── compare_sequential.py      # Benchmark comparison (RNN vs LSTM vs GRU)
├── 04_regularization_techniques/  # Module 4: Overfitting & Regularization Lab
│   ├── regularization_models.py   # Baseline, +Dropout, +BatchNorm, +EarlyStopping
│   └── ablation_study.py          # Empirical ablation study generating comparison curves
├── 05_transfer_learning_cnn/      # Module 5: Transfer Learning with Pretrained Models
│   ├── transfer_models.py         # MobileNetV2 / ResNet18 fine-tuning vs feature extractor
│   └── train_transfer.py          # Comparison between training from scratch vs transfer
├── 06_transformers_nlp/           # Module 6: Hugging Face Transformers
│   ├── nlp_triage.py              # Farmer inquiry classification and symptom extraction
│   └── pipeline_demo.py           # Pretrained Hugging Face pipeline demonstrations
├── 07_llm_agronomy_copilot/       # Module 7: LLM Diagnostic Synthesis
│   ├── llm_client.py              # Multi-provider client (OpenAI, Anthropic, Groq, Ollama, Fallback)
│   └── diagnostic_agent.py        # Multi-modal fusion agent (Vision + Telemetry + NLP)
├── 08_interactive_dashboard/      # Module 8: Unified Interactive Web Dashboard
│   └── app.py                     # Streamlit application with live demos of all 7 modules
├── data/                          # Sample datasets and generated assets
│   ├── sample_leaves/             # Sample images for testing (Healthy, Blight, Rust, Mildew)
│   └── telemetry/                 # Sample time-series sensor sequences
├── notebooks/                     # Interactive Jupyter Notebooks for each module
│   ├── 01_tensors_and_autograd.ipynb
│   ├── 02_cnn_classification.ipynb
│   ├── 03_rnn_lstm_gru.ipynb
│   ├── 04_regularization_ablation.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_huggingface_transformers.ipynb
│   └── 07_llm_copilot_integration.ipynb
├── tests/                         # Automated test suite
│   └── test_all_modules.py
├── requirements.txt               # Dependency specifications
└── README.md                      # Comprehensive documentation
```

---

## 🔬 Detailed Module Breakdown

### Module 1: Tensors, Autograd & Core Optimization
- Explores multidimensional tensors, indexing, slicing, broadcasting, and GPU/CPU device management.
- Implements automated differentiation (`autograd`) calculating gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$.
- Formulates a realistic agronomic optimization task: calibrating non-linear photosynthesis and nitrogen response parameters:
  $$\hat{Y} = A \cdot (1 - e^{-k \cdot N}) - c \cdot N^2$$
- Custom training loop with zeroing gradients, loss backward, optimizer step, and loss tracking.

### Module 2: CNN from Scratch (AgriLeafNet)
- Custom architectural design:
  - 3 Convolutional Blocks (Conv2D $3\times3$ with padding, ReLU, MaxPool2D $2\times2$)
  - Fully Connected classifier with Softmax
- Classifies 4 distinct agronomic states:
  1. `Healthy`
  2. `Bacterial_Blight`
  3. `Cereal_Rust`
  4. `Powdery_Mildew`
- Computes receptive fields, generates confusion matrices, and extracts intermediate activation feature maps.

### Module 3: Sequential Modeling (RNN vs LSTM vs GRU)
- Time-series sequence forecasting on multi-variate IoT telemetry (Soil Moisture %, Air Temperature, Relative Humidity, Solar Radiation, Soil pH).
- Architectures implemented:
  - **Vanilla RNN**: $\mathbf{h}_t = \tanh(\mathbf{W}_{ih}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b})$
  - **LSTM**: Input, forget, cell, and output gates resolving vanishing gradients.
  - **GRU**: Reset and update gates providing efficient sequence modeling.
- Evaluates MSE, MAE, convergence epochs, and inference latency across all three models.

### Module 4: Regularization Techniques
- Demonstrates the perils of overfitting on small or noisy real-world datasets.
- 4-way Empirical Ablation:
  1. **Unregularized Baseline**: High capacity, overfits quickly (Train loss $\approx 0$, Val loss spikes).
  2. **Dropout ($p=0.4$)**: Randomly deactivates neurons during training to prevent co-adaptation.
  3. **Batch Normalization**: Normalizes mini-batch activations, stabilizing gradients and speeding convergence.
  4. **Early Stopping**: Monitors validation loss with configurable patience and min-delta, automatically restoring the best model weights.
- Generates comparative training/validation loss curves.

### Module 5: Transfer Learning (Pretrained CNNs)
- Leverages industry standard pretrained backbones (`MobileNetV2` and `ResNet18`).
- Explores two distinct regimes:
  1. **Feature Extraction**: Freezes the feature extraction backbone (`param.requires_grad = False`), replacing and training only the final linear classification head.
  2. **Gradual Fine-Tuning**: Selectively unfreezes deep convolutional layers with a reduced learning rate ($\times 0.1$).
- Demonstrates rapid convergence (95%+ accuracy in 3 epochs) vs training from scratch.

### Module 6: Hugging Face Transformers
- Integrates Hugging Face `transformers` pipeline and model architectures.
- Performs intent classification, symptom triage, and urgency level detection from farmer field text queries (e.g., *"My tomato leaves have yellow spots and are curling up rapidly after heavy rain"*).
- Demonstrates tokenization, attention masks, model embeddings, and pipeline abstractions.

### Module 7: LLM Diagnostics & Multi-Modal Copilot
- Synthesizes findings across Vision (Leaf Disease), Telemetry (Soil & Weather Forecast), and Farmer Inquiries into a structured **Agronomic Diagnostic & Remediation Report**.
- Supports multiple providers:
  - OpenAI (`gpt-4o`, `gpt-4o-mini`)
  - Anthropic (`claude-3-5-sonnet`)
  - Groq (`llama-3.3-70b-versatile`)
  - Ollama (Local open-source models like `llama3`, `mistral`, `phi3`)
  - Offline Deterministic Expert Fallback (guarantees 100% operation without API keys).

---

## 👥 Contributors & Academic Acknowledgments
Developed for **SIT-N Hack-O-Week, Batch 2024-28 (5th Semester)**.
Designed to adhere strictly to deep learning best practices, reproducible research, and modular software engineering.
