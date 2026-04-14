# XAI Healthcare Decision Support System

A production-grade seminar prototype for healthcare triage and prediction with Explainable AI (XAI).

## Project Structure

- `dashboard.html`: Main interaction page with chatbot and triage logic.
- `performance.html`: Detailed model evaluation metrics and visualizations.
- `comparison.html`: Benchmarking against existing systems and techniques.
- `style.css`: Shared clinical design system (Light/Dark mode).
- `model/`:
    - `train_model.py`: Python script for data processing and training.
    - `metrics.json`: Exported performance data.
    - `feature_columns.json`: List of all symptom features.
- `data/`: CSV datasets downloaded from source.

## Setup & Usage

### 1. Model Training (Optional for Dashboard)
The dashboard uses hardcoded client-side logic for the prototype, but you can run the training pipeline to verify metrics:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python model/train_model.py
```

### 2. View Dashboard
Simply open `index.html` or `dashboard.html` in any modern web browser.

## Features
- **Clinical Design**: Minimalist SaaS aesthetic using IBM Plex Sans.
- **Multilevel Triage**: 3-zone severity scoring (Green, Yellow, Red).
- **Explainable AI**: SHAP-style importance bars for symptoms.
- **Decision Support**: Age-specific medication recommendations for home care.
- **Interactive Analytics**: 6+ detailed charts for model performance and comparison.

---
*Educational prototype only — not a substitute for professional medical advice.*
