# Cardiovascular Disease Risk Intelligence Platform

Production-ready, modular Machine Learning and Clinical Decision Support System designed for cardiovascular risk stratification and personalized evidence-based medical recommendations.

---

## 🏗️ Architecture & Project Structure

```
heart-disease-risk-analysis/
├── data/
│   ├── raw/heart_disease_raw.csv           # Raw patient records (or auto-generated synthetic benchmark)
│   └── processed/heart_disease_clean.csv   # Cleaned, validated, and normalized dataset
├── src/
│   ├── __init__.py
│   ├── config.py                           # Paths, clinical thresholds, schema & hyperparameter config
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── loader.py                       # Data ingestion, schema validation & synthetic generator
│   │   └── preprocessor.py                 # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                        # Pipeline (Preprocessor -> PCA 95% -> SVC RBF) + GridSearchCV
│   │   └── evaluate.py                     # Performance evaluation & metric serialization
│   ├── services/
│   │   ├── __init__.py
│   │   ├── predictor.py                    # Production inference engine (predict & predict_proba)
│   │   └── recommender.py                  # Clinical guideline evaluation (ACC/AHA thresholds)
│   └── database/
│       ├── __init__.py
│       └── db_handler.py                   # SQLite audit trail integration (patients.db)
├── artifacts/
│   ├── svm_pca_pipeline.joblib             # Serialized trained scikit-learn pipeline
│   └── model_metrics.json                  # Logged evaluation metrics & confusion matrix
├── app/
│   ├── streamlit_app.py                    # Clinical interactive web dashboard
│   └── components/
│       └── ui_widgets.py                   # Custom visual widgets, risk badges & audit trail view
├── tests/
│   ├── test_data_pipeline.py               # Unit tests for loader, schema, and preprocessor
│   └── test_prediction.py                  # Integration tests for inference and probability constraints
├── requirements.txt                        # Pinned dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Key Features

1. **Leak-Free Scikit-Learn Pipeline**:
   - `StandardScaler` for continuous numerical features.
   - `OneHotEncoder` for categorical factors (`Gender`).
   - `PCA(n_components=0.95)` dimensionality reduction preserving 95% of explained variance.
   - `SVC(kernel='rbf', probability=True, class_weight='balanced')` calibrated for imbalanced medical diagnosis.
2. **5-Fold Stratified Cross-Validation & Grid Search**:
   - Hyperparameter optimization across $C$ and $\gamma$ parameters using `GridSearchCV(scoring='roc_auc')`.
3. **Automated Data Validation & Synthetic Benchmark**:
   - Generates a 300-row realistic synthetic clinical benchmark dataset if `heart_disease_raw.csv` is missing or empty.
4. **Clinical Decision Support Guidelines**:
   - Evaluates resting blood pressure ($\ge 130$ mmHg), serum cholesterol ($\ge 200$ mg/dL), fasting glucose ($\ge 126$ mg/dL), BMI ($\ge 25.0$), and resting ECG anomalies.
5. **SQLite Patient Audit Logging**:
   - Every inference run logs patient biomarkers, risk probability, classification, and clinical notes to `patients.db`.
6. **Dark-Mode Friendly Streamlit UI**:
   - Interactive intake form, real-time risk gauge, clinical alert banners, and expandable historical audit log view.

---

## 🛠️ Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model Pipeline
Execute the end-to-end data pipeline and model training script:
```bash
python -m src.models.train
```
This produces:
- `artifacts/svm_pca_pipeline.joblib`
- `artifacts/model_metrics.json`
- `data/processed/heart_disease_clean.csv`

### 3. Run Automated Tests
```bash
pytest -v
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
