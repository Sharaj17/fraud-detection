# 🕵️‍♂️ Fraud Detection (Credit Card) — Day 01

This project is part of my **Data Science Portfolio Framework**.  
Day 01 builds a **reproducible, production-friendly baseline** for credit card fraud detection:

- Scripted **data ingestion** (no manual downloads)
- Quick **EDA** focused on class imbalance
- **Logistic Regression** baseline (interpretable, calibrated)
- **MLflow** experiment tracking with metrics & artifacts

> The goal today is **learning-by-doing** with clean, reproducible steps you can rerun anytime.

---

## 📦 Tech Stack

- Python 3.12, `pandas`, `numpy`, `scikit-learn`, `mlflow`, `joblib`
- (Later) `fastapi`, `uvicorn`, `pytest`, Docker, Kubernetes

---

## 📁 Repo Layout (today)

fraud-detection/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│ ├─ .gitkeep # keep folder tracked; raw data is local-only
│ └─ raw/ # creditcard.csv will be saved here (ignored)
├─ notebooks/
│ ├─ 01_eda_quickstart.py
│ └─ models/ # trained models (.pkl) ⇒ ignored by git
└─ src/
├─ init.py
└─ fraud/
├─ init.py
├─ download_data.py # pulls dataset from OpenML
└─ train_baseline.py # trains & logs a baseline model (LogReg)

**Why this structure?**  
It separates **code / data / artifacts**, which is exactly how production teams keep repos clean and reproducible.

---

## 🧰 Setup (once)

```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

requirements.txt (what I use today):

pandas>=2.2,<2.3
numpy>=2.0,<2.4
scikit-learn>=1.5,<1.7
mlflow>=2.14
joblib>=1.3
matplotlib>=3.8


⬇️ Data Ingestion (scripted)

File: src/fraud/download_data.py

Pulls the well-known Credit Card Fraud dataset from OpenML (same as Kaggle but no login).

Saves it locally to data/raw/creditcard.csv so runs are reproducible.

Run it:

python src/fraud/download_data.py


What it does:

Uses fetch_openml(name="creditcard", version=1, as_frame=True)

Writes a CSV locally so all models/EDA can read the same snapshot

Prints shape & path saved

This is how real teams ingest data: programmatically and repeatably, not by hand.

🔎 Quick EDA (learn the data you’re modeling)

File: notebooks/01_eda_quickstart.py

Run:

python notebooks/01_eda_quickstart.py


What to look for:

Shape: ~284,807 rows × 31 columns

Target: Class (0 = legit, 1 = fraud)

Class imbalance: fraud ≪ legit (heavily imbalanced)

Features: V1..V28 (PCA components), plus Amount, Time

Nulls: typically none in this dataset

Key learning: Imbalance dominates fraud detection. Choose metrics accordingly (ROC-AUC, PR-AUC), and handle imbalance (class_weight='balanced', threshold tuning, etc.).

🤖 Baseline Model (Logistic Regression)

File: src/fraud/train_baseline.py

Core ideas:

Stratified train/val/test splits (preserve class ratios)

Logistic Regression with class_weight='balanced' (critical for fraud rarity)

Evaluate with ROC-AUC and Average Precision (PR-AUC)

Log to MLflow: params, metrics, model artifact

Run:

python src/fraud/train_baseline.py

Why Logistic Regression (Day 01)?

Interpretable: coefficients explain how features affect risk

Calibrated: predict_proba gives meaningful probabilities

Lightweight: fast to train & deploy; perfect baseline

We’ll compare against non-linear models (RF/XGBoost) later, but LogReg is the right place to start for regulated/operational settings.

🧪 Metrics You’ll See (typical)
Metric	Why it matters
ROC-AUC (val/test)	Ranking ability across thresholds; robust to imbalance
Average Precision (val/test)	Area under Precision–Recall curve; more informative at very low positive rates
Classification Report	Precision/Recall/F1 at a 0.5 threshold (we’ll tune thresholds later)

Typical baselines:

ROC-AUC: ~0.95 (LogReg), ~0.97 (RF)

AP (PR-AUC): ~0.80 (LogReg), ~0.85 (RF)

(Your exact numbers may vary slightly.)

🧾 MLflow (experiment tracking)

Start the UI (optional but recommended):

mlflow ui
# open http://127.0.0.1:5000


You’ll see a run for today’s training with:

Parameters (e.g., C, penalty, solver)

Metrics (e.g., roc_auc_test, ap_val)

Artifacts (model file)

Pro tip: name your experiment and runs for tidy tracking:

mlflow.set_experiment("fraud_detection_baselines")
with mlflow.start_run(run_name="logreg_baseline_creditcard"):
    ...

🔒 .gitignore (keep repo light & professional)

We do not push data, models, or MLflow runs to GitHub:

# Data
data/
!data/.gitkeep

# Models / artifacts
notebooks/models/

# MLflow local tracking
mlruns/

# venv & caches
venv/
__pycache__/
*.pyc

🧠 What I Learned Today (Day 01)

Scripted data ingestion beats manual downloads (reproducibility).

Fraud is heavily imbalanced → pick the right metrics.

Logistic Regression is a powerful baseline:

interpretable coefficients,

class-weighted to handle imbalance,

calibrated probabilities for business thresholds.

MLflow is “Git for experiments” — log everything you want to remember.

🚀 What’s Next (Day 02+)

Build a FastAPI prediction service for the trained model.

Add pytest tests for valid/invalid payloads.

Package with Docker, then deploy to Kubernetes (reusing the framework you’ve built).

Iterate on modeling (threshold tuning, cost-sensitive evaluation, feature engineering).


---

If you want, I can also draft **Day 02** README sections (API + tests) now, but since you asked for Day 01, this keeps things focused and clean. When you’re ready, we’ll move to **inference API** for the fraud model next.
::contentReference[oaicite:0]{index=0}