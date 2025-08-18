# OmniForge 🤖

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A versatile AI framework designed to tackle a variety of machine learning tasks, from complex image classification to nuanced tabular data regression. This repository serves as a demonstration of end-to-end project implementation, featuring reusable, modular code for efficient model development.

---

## 📋 Overview

OmniForge is built around two core, specialized frameworks:

* **`ClassiForge.py`**: A powerful and flexible pipeline for **image classification**. It leverages pre-trained models (like ResNet and Vision Transformers) to achieve state-of-the-art performance on diverse image datasets.
* **`CognitionForge.py`**: A complete framework for **tabular regression** tasks. It includes a full preprocessing pipeline using Scikit-learn and a custom PyTorch neural network to predict continuous values.

This project showcases the ability to build and train distinct models for different problem domains, all while maintaining a clean and organized codebase.

---

## 🚀 Projects & Performance

This repository contains several projects built on the OmniForge frameworks. The performance metrics below were achieved after training on the respective datasets.

| Project Name | Task | Framework | Best Performance |
| :--- | :--- | :--- | :--- |
| **FlutterFrame** | Moth & Butterfly Classification | `ClassiForge` | **98.40%** Accuracy |
| **Decksentience** | Playing Card Classification | `ClassiForge` | **97.36%** Accuracy |
| **AthloScope** | 100 Sports Classification | `ClassiForge` | **97.20%** Accuracy |
| **BreedNova** | 70 Dog Breeds Classification | `ClassiForge` | **96.86%** Accuracy |
| **CognitionForge**| Teen Addiction Level Prediction | `CognitionForge` | **0.178** MAE |

---

## 🛠️ Getting Started

Follow these instructions to set up the project environment and run the models on your local machine.

### Prerequisites

* **Git**
* **Python 3.10** or newer
* **NVIDIA GPU** with CUDA support is highly recommended for training.

> **Note**: This project was developed and tested on **Ubuntu 22.04 LTS**. While it is expected to be cross-platform, minor adjustments might be needed for other operating systems.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TheVictor777/OmniForge.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd OmniForge
    ```

3.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it (on Linux/macOS)
    source venv/bin/activate
    ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
---
## 📚 Datasets

The datasets are not included in this repository and must be downloaded manually.

1.  In the root of the project, create a folder named `Datasets`.
2.  Download the required datasets from the links below.
3.  Unzip them and ensure they are placed inside the `Datasets` folder with the **exact folder names** specified in the table.

| Project | Download Link | Required Folder Structure |
| :--- | :--- | :--- |
| **CognitionForge** | [Kaggle Link](https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction) | `Datasets/Teen Smartphone Usage and Addiction Impact Dataset/` |
| **FlutterFrame** | [Kaggle Link](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species) | `Datasets/Butterfly & Moths Image Classification 100 species/` |
| **AthloScope** | [Kaggle Link](https://www.kaggle.com/datasets/gpiosenka/sports-classification) | `Datasets/100 Sports Image Classification/` |
| **Decksentience** | [Kaggle Link](https://www.kaggle.com/datasets/86dcbfae1396038cba359d58e258915afd32de7845fd29ef6a06158f80d3cce8) | `Datasets/Cards Image Dataset-Classification/` |
| **BreedNova** | [Kaggle Link](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set) | `Datasets/70 Dog Breeds-Image Data Set/` |

---
## 💻 Usage

With the environment and datasets set up, you can now train, evaluate, or run predictions.

### CognitionForge (Tabular Regression)

* **To train the model:**
    ```bash
    python3 CognitionForge.py --mode train
    ```

* **To evaluate the trained model:**
    ```bash
    python3 CognitionForge.py --mode evaluate
    ```

* **To make a prediction:**
    ```bash
    python3 CognitionForge.py --mode predict
    ```

### ClassiForge (Image Classification)

* **To train the dog breed classifier (`BreedNova`):**
    ```bash
    python3 BreedNova.py
    ```

* **To train the sports classifier (`AthloScope`):**
    ```bash
    python3 AthloScope.py
    ```
---
## 📂 Project Structure

````

OmniForge/
│
├── .gitattributes          \# Configures Git LFS for large files
├── .gitignore              \# Specifies files for Git to ignore
│
├── AthloScope.py           \# Project: Sports classification
├── BreedNova.py            \# Project: Dog breed classification
├── ClassiForge.py          \# Core Framework: Image Classification
├── CognitionForge.py       \# Core Framework: Tabular Regression
├── Decksentience.py        \# Project: Playing card classification
├── FlutterFrame.py         \# Project: Butterfly/moth classification
│
├── \*.pth                   \# Trained model files (handled by LFS)
├── preprocessor.joblib     \# Preprocessor file (handled by LFS)
│
├── requirements.txt        \# Project dependencies for pip
├── README.md               \# This file
└── utils.py                \# Utility functions

````
---

## 📄 License

This project is distributed under the MIT License.

---

## 📫 Contact

Victor - [@TheVictor777](https://github.com/TheVictor777)
