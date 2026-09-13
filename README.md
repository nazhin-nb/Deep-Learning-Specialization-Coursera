# Deep Learning Specialization — Coursera & DeepLearning.AI
### Master Deep Learning, Neural Network Architectures, Optimization, and Engineering
#### By Dr. Andrew Ng — DeepLearning.AI & Coursera

[![Specialization](https://img.shields.io/badge/Coursera-Deep%20Learning%20Specialization-0056D2.svg?logo=coursera&logoColor=white)](https://www.coursera.org/specializations/deep-learning)
[![Organization](https://img.shields.io/badge/Organization-DeepLearning.AI-FF6F00.svg)](https://www.deeplearning.ai/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![TensorFlow](https://img.shields.io/badge/Library-TensorFlow%202.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Courses%201%20%26%202%20Completed-brightgreen.svg)](#-completed-courses)
[![Tests](https://img.shields.io/badge/Tests-All%20Passed-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 📌 Repository Overview

This repository contains completed, fully-benchmarked programming assignments, modular implementations, and deep mathematical foundations for the **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)** created by **Dr. Andrew Ng** and offered through **[DeepLearning.AI](https://www.deeplearning.ai/)** on **Coursera**.

The specialization is designed to teach machine learning practitioners how to construct, optimize, tune, and deploy deep neural network models. The code progression in this repository follows a principled two-stage pedagogical philosophy:
1. **Foundations from Scratch (Pure Python & NumPy)**: Deconstructing forward propagation, backpropagation, cost functions, gradient updates, initialization, and regularization without reliance on black-box frameworks.
2. **Modern Industrial Tooling (TensorFlow 2.x)**: Implementing custom training loops, `tf.GradientTape` automatic differentiation, and GPU-optimized data streaming pipelines with `tf.data`.

---

## 🗺️ Completed Courses

| Course | Title | Status | Primary Focus | Frameworks & Tools |
|:---:|:---|:---:|:---|:---|
| **Course 1** | [**Neural Networks and Deep Learning**](./Course1-Neural-Networks-and-Deep-Learning) | ✅ **Completed** (100%) | Vectorization, Logistic Regression, Shallow NNs, Modular $L$-Layer Deep NN Engine, Image Classification | Python, NumPy, Matplotlib |
| **Course 2** | [**Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization**](./Course2-Neural-Networks-and-Deep-Learning) | ✅ **Completed** (100%) | Weight Initialization, $L_2$ Regularization, Inverted Dropout, Gradient Checking, Mini-Batch GD, Momentum, Adam, LR Decay, TensorFlow 2.x | Python, NumPy, TensorFlow 2.x, `tf.data` |

---

## 🔬 Course-by-Course Highlights

### 🔹 [Course 1: Neural Networks and Deep Learning](./Course1-Neural-Networks-and-Deep-Learning)
> *Comprehensive guide available in [Course 1 README](./Course1-Neural-Networks-and-Deep-Learning/README.md).*

* **Core Objectives**: Build vectorization fundamentals and a modular deep learning engine from scratch with pure NumPy.
* **Key Implementations**:
  - Vectorized mathematical operations: Numerically stable sigmoid, row-wise softmax, $L_1$ and $L_2$ losses.
  - **Single-Neuron Logistic Regression** on Cat vs. Non-Cat dataset (**70.0% test accuracy**).
  - **2-Layer Planar Classifier** with $\tanh$ hidden units and random symmetry-breaking initialization (**90.0% accuracy**).
  - **Modular $L$-Layer Deep Neural Network**: General-purpose engine with cached linear and activation layers (`linear_forward`, `linear_activation_forward`, `linear_backward`, `update_parameters`).
  - **Cat Classifier Benchmark**: A 4-layer deep architecture (`[12288, 20, 7, 5, 1]`) achieving **80.0% test accuracy** (+10% over logistic regression).

---

### 🔹 [Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](./Course2-Neural-Networks-and-Deep-Learning)
> *Comprehensive guide available in [Course 2 README](./Course2-Neural-Networks-and-Deep-Learning/README.md).*

* **Core Objectives**: Master practical deep learning engineering to eliminate vanishing/exploding gradients, combat overfitting, accelerate convergence, and transition to modern frameworks.
* **Key Implementations**:
  - **Weight Initialization**: Compared Zero, Large Random ($W \times 10$), and **He Initialization** ($W \sim \mathcal{N}(0, 1)\sqrt{2/n^{[l-1]}}$). He initialization boosted planar classification to **96.0% test accuracy**.
  - **Regularization**: Implemented **$L_2$ Regularization (Weight Decay)** and **Inverted Dropout** ($p = 0.86$), driving test accuracy on the French football dataset to **95.0%**.
  - **Gradient Checking**: Built $1\text{D}$ and $N\text{D}$ numerical backpropagation verifiers using finite differences with precision tolerance $\varepsilon < 10^{-7}$ ($\text{diff} \approx 7.8 \times 10^{-11}$).
  - **Advanced Optimization**: Coded **Mini-Batch GD**, **Momentum**, and **Adam** from scratch. Benchmarked with learning rate decay schedules to achieve **95.3% accuracy** on complex non-linear boundaries.
  - **TensorFlow 2.x Introduction**: Built end-to-end multi-class image classifier on the **SIGNS dataset** (6 gesture classes) using `tf.data.Dataset` caching/prefetching, `tf.GradientTape` automatic differentiation, and Adam optimization.

---

## 📊 Performance & Accuracy Benchmark Summary

| Course | Model / Experiment | Architecture | Optimizer / Tuning | Train Acc | Test Acc |
|:---:|:---|:---:|:---|:---:|:---:|
| **C1** | Logistic Regression (W2A2) | `[12288, 1]` | Batch GD ($\alpha=0.005$) | 99.0% | **70.0%** |
| **C1** | Shallow Planar NN (W3A1) | `[2, 4, 1]` | Batch GD ($\alpha=1.2$) | 90.0% | **90.0%** |
| **C1** | Deep 4-Layer NN (W4A2) | `[12288, 20, 7, 5, 1]` | Batch GD ($\alpha=0.0075$) | 98.6% | **80.0%** |
| **C2** | Zero Initialization (W1A1) | `[2, 10, 5, 1]` | Batch GD (Zero weights) | 50.0% | **50.0%** |
| **C2** | He Initialization (W1A1) | `[2, 10, 5, 1]` | Batch GD (He calibrated) | 99.3% | **96.0%** |
| **C2** | Non-Regularized Baseline (W1A2) | `[2, 20, 3, 1]` | Batch GD (Overfitted) | 94.8% | **91.5%** |
| **C2** | $L_2$ Regularized Model (W1A2) | `[2, 20, 3, 1]` | Batch GD ($\lambda=0.7$) | 93.8% | **93.0%** |
| **C2** | Inverted Dropout Model (W1A2) | `[2, 20, 3, 1]` | Batch GD ($\text{keep}=0.86$) | 92.9% | **95.0%** |
| **C2** | Adam Optimizer (W2A1) | `[2, 5, 2, 1]` | Mini-Batch Adam | 94.3% | **94.3%** |
| **C2** | Momentum + LR Decay (W2A1) | `[2, 5, 2, 1]` | Mini-Batch Momentum + Decay | 95.3% | **95.3%** |
| **C2** | TensorFlow SIGNS Classifier (W3A1) | `[12288, 128, 64, 6]` | Adam (`GradientTape`, 100 ep) | 75.4% | **69.2%** |

---

## 🛠️ Tech Stack & Dependencies

The implementations in this repository utilize:
* **Languages**: Python 3.8+
* **Core Scientific Computing**: NumPy, SciPy
* **Data Visualization**: Matplotlib
* **Data Formats & Storage**: HDF5 (`h5py`), Pickle
* **Image Processing**: Pillow (PIL)
* **Deep Learning Frameworks**: TensorFlow 2.x (`tf.data`, `tf.keras`, `tf.GradientTape`)
* **Environment**: Jupyter Notebook / JupyterLab

---

## 📁 Repository Structure

```text
Deep-Learning-Specialization-Coursera/
├── README.md                                             # Root Specialization Hub (This file)
├── LICENSE                                               # MIT License
│
├── Course1-Neural-Networks-and-Deep-Learning/            # Course 1: Foundations
│   ├── README.md                                         # Detailed Course 1 Documentation
│   ├── W2A1/                                             # Week 2: Python Basics with NumPy
│   ├── W2A2/                                             # Week 2: Logistic Regression as a NN
│   ├── W3A1/                                             # Week 3: Planar Data Classification (1-Hidden Layer)
│   ├── W4A1/                                             # Week 4: Step-by-Step Modular Deep NN Engine
│   └── W4A2/                                             # Week 4: Deep NN Image Classification (Application)
│
└── Course2-Neural-Networks-and-Deep-Learning/            # Course 2: Improving Deep NNs
    ├── README.md                                         # Detailed Course 2 Documentation
    ├── Week1/                                            # Week 1: Practical Aspects of DL
    │   ├── W1A1/                                         # Weight Initialization (Zero, Random, He)
    │   ├── W1A2/                                         # Regularization (L2 & Inverted Dropout)
    │   └── W1A3/                                         # Numerical Gradient Checking (1D & ND)
    ├── Week2/                                            # Week 2: Optimization Algorithms
    │   └── W2A1/                                         # Mini-batch, Momentum, Adam, LR Decay
    └── Week3/                                            # Week 3: Introduction to TensorFlow
        └── W3A1/                                         # TF 2.x, GradientTape & SIGNS Classifier
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nazhin-nb/Deep-Learning-Specialization-Coursera.git
cd Deep-Learning-Specialization-Coursera
```

### 2. Set Up a Virtual Environment
```bash
# Using conda
conda create -n dls python=3.10
conda activate dls

# Or using venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy matplotlib h5py scipy Pillow tensorflow
```

### 4. Launch Jupyter
```bash
jupyter notebook
```
Navigate to any course notebook and run all cells sequentially. All unit tests and model evaluations pass completely.

---

## 🎓 Acknowledgements & Credits

* **Course Instructor**: [Dr. Andrew Ng](https://www.andrewng.org/)
* **Instructional Team**: DeepLearning.AI
* **Platform**: [Coursera](https://www.coursera.org/specializations/deep-learning)
* **Specialization**: Deep Learning Specialization

---

## 📄 License

This repository is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.