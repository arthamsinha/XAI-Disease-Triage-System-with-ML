# XAI Healthcare Decision Support System

A production-grade seminar prototype for healthcare triage and prediction with Explainable AI (XAI). This system integrates a professional Python-based Machine Learning pipeline with a responsive web dashboard to provide real-time, data-driven diagnostic support.

## Key Features
- **Integrated Pipeline**: The dashboard logic is directly synced with the Python ML model via an automated metadata export.
- **Explainable AI (XAI)**: Provides clear, symptom-based explanations for every prediction using evidence-matching logic.
- **Layman-Friendly Care**: Diagnostic reports include clear precautions and care plans translated into plain language.
- **Genuine Analytics**: All performance scores (accuracy, precision, recall) are generated from actual training runs on the dataset.

## Project Structure

- `dashboard.html`: Main interaction page with chatbot and triage logic (loads data dynamically).
- `data/master_clinical_knowledge.csv`: The **Source of Truth** for all disease descriptions, medications, and precautions.
- `model/`:
    - `train_model.py`: Automates model training and clinical knowledge consolidation.
    - `Model_Analysis_Proof.ipynb`: A technical walkthrough/evidence of model training and XAI logic.
    - `metadata.json`: The consolidated package used by the web dashboard.
- `data/`: CSV datasets used for training.

## Setup & Usage

### 1. Model Training & Data Sync
The dashboard is powered by the Python training script. Run this to sync your CSV edits and update the model:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training & knowledge sync
python model/train_model.py
```

### 2. View Dashboard
Simply open `dashboard.html` in any modern web browser once the metadata is generated.

## Acknowledgments & Data Source

The underlying medical symptom dataset used in this project is sourced from:
- **Soham V Sonar**: [Disease Prediction and Medical Recommendation System](https://github.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System)

---
*Educational prototype only — not a substitute for professional medical advice.*
