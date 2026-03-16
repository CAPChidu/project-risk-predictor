# Project Risk Predictor

> **Roles targeted:** Junior Project Manager | Business Analyst | Data Analyst

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Business Problem

Project managers face a critical question at every stage: *is this project on track to succeed, or is it heading for failure?* Traditional risk management relies on manual assessments that are slow, subjective, and inconsistent. This tool replaces that with a data-driven, ML-powered risk prediction system.

---

## What I Built

An ML-powered risk prediction system that analyzes historical project data to forecast project outcomes, identify risk factors, and give PMs actionable early-warning signals.

| Component | Description |
|---|---|
| Risk Classifier | Predicts project success/failure probability using 15+ features |
| Feature Importance Analysis | Identifies which risk factors matter most (scope creep, budget, team size) |
| Early Warning System | Flags at-risk projects at Week 4, 8, and 12 checkpoints |
| Risk Heatmap | Visual matrix showing risk concentration across portfolio |
| Scenario Analysis | What-if modelling for different mitigation strategies |

---

## Key Findings & Business Impact

- Model achieves **82% accuracy** and **0.87 AUC** on test data
- Top 3 risk predictors: **scope change frequency**, **stakeholder engagement score**, and **budget variance %**
- Early warning at Week 4 gives PMs **8 weeks more lead time** to intervene vs. reactive management
- Portfolio-level risk heatmap enables PMO to prioritize attention across 10+ concurrent projects
- Scenario analysis shows mitigation strategies can reduce failure probability by up to **35%**

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python 3.9+ |
| Machine Learning | Scikit-learn (Random Forest, Logistic Regression, XGBoost) |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Model Evaluation | Cross-validation, ROC-AUC, confusion matrix |

---

## Project Structure

```
project-risk-predictor/
├── src/
│   ├── data_prep/               # Feature engineering and preprocessing
│   ├── models/                  # ML model training and evaluation
│   ├── prediction/              # Real-time prediction pipeline
│   └── visualization/           # Risk dashboards and heatmaps
├── data/                        # Historical project datasets
├── notebooks/                   # Exploratory analysis notebooks
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ChidghanaH/project-risk-predictor.git
cd project-risk-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/models/train.py

# 5. Run predictions
python src/prediction/predict.py
```

---

## Skills Demonstrated

- Supervised machine learning: classification, ensemble methods, model evaluation
- Feature engineering for PM domain: scope, schedule, budget, stakeholder variables
- Business Analysis: translating model outputs into PM-actionable risk recommendations
- Data visualization: risk heatmaps, feature importance plots, ROC curves
- End-to-end ML project: data prep → training → evaluation → deployment-ready pipeline

---

## Author

**Chidghana Hemantharaju** — MSc Business Analytics | Munich, Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/chidghana-hemantharaju/)
