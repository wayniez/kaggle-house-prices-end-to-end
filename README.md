# 🏠 Kaggle House Prices — End-to-End ML Project

An end-to-end machine learning project for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition. The project covers the full ML pipeline — from exploratory data analysis and feature engineering to model training and an interactive Streamlit web application for real-time price prediction.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Dev Container](#dev-container)

---

## Overview

The goal is to predict residential home sale prices in Ames, Iowa based on 79 features describing every aspect of the property. The project is structured as a realistic end-to-end workflow rather than a single notebook submission.

**Key highlights:**
- Full EDA and feature engineering pipeline
- Ensemble of XGBoost, LightGBM, and ElasticNet models
- Interactive Streamlit app for generating predictions
- Dev container setup for reproducible development environments

---

## Project Structure

```
kaggle-house-prices-end-to-end/
├── app/                    # Streamlit web application
├── .devcontainer/          # VS Code Dev Container configuration
└── README.md
```

---

## ML Pipeline

1. **Exploratory Data Analysis** — distribution of target variable, correlation analysis, missing value inspection
2. **Feature Engineering** — handling missing values, encoding categorical features, skewness correction via log-transform
3. **Modeling** — training and tuning three models:
   - `XGBoost` — gradient boosting with tree-based learners
   - `LightGBM` — fast gradient boosting optimized for large datasets
   - `ElasticNet` — regularized linear regression (L1 + L2 penalty)
4. **Ensemble** — blending model predictions for improved generalization
5. **Deployment** — Streamlit app for interactive inference

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Modeling | `scikit-learn`, `xgboost`, `lightgbm` |
| App | `streamlit` |
| Environment | Dev Container (VS Code) |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/wayniez/kaggle-house-prices-end-to-end.git
cd kaggle-house-prices-end-to-end
pip install -r requirements.txt
```

### Data

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place the files in a `data/` directory:

```
data/
├── train.csv
└── test.csv
```

---

## Running the App

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`. Enter house features to get an instant price prediction.

---

## Dev Container

This project includes a `.devcontainer` configuration for VS Code. To use it:

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
2. Open the repository in VS Code
3. Click **Reopen in Container** when prompted

All dependencies will be installed automatically inside the container.



