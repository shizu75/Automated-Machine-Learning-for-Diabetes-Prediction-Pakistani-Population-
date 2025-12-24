# Automated Machine Learning for Diabetes Prediction (Pakistani Population)

## Project Overview

This repository presents an **end-to-end machine learning pipeline for diabetes prediction** using a **Pakistani clinical dataset**, with a strong emphasis on **statistical rigor, feature diagnostics, and automated model selection**.  
The work is structured in a **research-oriented manner**, suitable for **PhD-level portfolios**, emphasizing reproducibility, interpretability, and methodological soundness.

The project combines:
- Exploratory Data Analysis (EDA)
- Multicollinearity diagnostics
- Feature selection
- Automated Machine Learning (AutoML) using **PyCaret**
- Model evaluation and interpretation

---

## Scientific Motivation

Diabetes is a major public health challenge in South Asia. Predictive modeling on region-specific datasets is crucial because:

- Clinical feature distributions vary by population
- Risk factors differ across demographics
- Locally trained models improve reliability and fairness

This study focuses on **data-driven, statistically validated model selection** rather than manual trial-and-error modeling, aligning with modern **evidence-based ML research practices**.

---

## Dataset Description

- **Dataset**: Pakistani Diabetes Dataset
- **Target Variable**: `Outcome` (Binary: Diabetic / Non-Diabetic)
- **Features**: Clinical and biochemical measurements
- **Preprocessing**:
  - Duplicate removal
  - Feature correlation analysis
  - Multicollinearity assessment

---

## Exploratory Data Analysis (EDA)

### Outcome Distribution
- Histogram visualization used to inspect class balance

### Correlation Analysis
- Pearson correlation heatmaps generated to:
  - Identify redundant variables
  - Guide feature removal
- Highly correlated features (`B.S.R`, `Dur`) removed to reduce redundancy

---

## Feature Diagnostics

### Multicollinearity Analysis
- **Variance Inflation Factor (VIF)** and **Tolerance** computed after standardization
- Ensures:
  - Stable coefficient estimation
  - Improved generalization
  - Reduced overfitting risk

This step is critical for medical ML studies where interpretability and statistical validity matter.

---

## Automated Machine Learning (AutoML)

The modeling stage is powered by **PyCaret (Classification Module)**:

### Pipeline Includes
- Automated preprocessing
- Cross-validation
- Model benchmarking
- Hyperparameter tuning

### Key Steps
- `setup()` for experiment initialization
- `compare_models()` for baseline benchmarking
- `automl()` for optimal model selection (Accuracy-optimized)
- `tune_model()` for hyperparameter refinement

This approach ensures **objective and reproducible model selection**.

---

## Model Evaluation & Interpretation

The final tuned model is evaluated using:

- Confusion Matrix
- ROC–AUC Curve
- Decision Boundary Visualization
- Interactive evaluation dashboard

Predictions are generated on the full dataset to inspect classification behavior.

---

## Key Outcomes

- Robust diabetes prediction model tailored to Pakistani clinical data
- Removal of multicollinearity improves model stability
- AutoML enables transparent, bias-minimized model selection
- Strong balance between performance and interpretability

---

## Research & PhD Relevance

This project demonstrates:

- Statistical awareness (correlation, VIF, tolerance)
- Responsible feature selection for healthcare ML
- Modern AutoML workflows used in applied research
- Reproducible and extensible experimental design

It is well-suited as:
- A **flagship applied ML project**
- A foundation for **clinical decision-support research**
- A baseline for future work in **biomedical AI**

---

## Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Statsmodels
- PyCaret (Classification)

---

## Ethical & Academic Use

This repository is intended for **research and academic purposes only**.  
If reused in publications or derivative research, appropriate citation and ethical compliance are expected.

---

## Author Note
Soban Saeed.
