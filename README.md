# Cardiovascular Disease Risk Intelligence Platform

A modular Machine Learning, FastAPI Backend, and Clinical Decision Support research prototype designed for cardiovascular risk stratification, patient profile management, and personalized evidence-based medical recommendations.

> ⚠️ **Research Prototype Disclaimer**: This system is an experimental computational research prototype evaluated using synthetic benchmark data. Its predictions, probability calibrations, and automated recommendations are **not clinically validated** and must not be used for direct medical diagnosis, prognosis, or clinical treatment planning.

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
│   │   ├── loader.py                       # Data ingestion, schema validation & synthetic generator
│   │   └── preprocessor.py                 # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── models/
│   │   ├── calibration.py                  # Probability calibration (Platt Sigmoid / Isotonic)
│   │   ├── evaluate.py                     # Performance evaluation & metric serialization
│   │   ├── model_comparison.py             # Multi-model CV comparison (SVM, RF, XGBoost)
│   │   ├── model_factory.py                # Pipeline construction factory
│   │   └── train.py                        # Model training and artifact serialization
│   ├── services/
│   │   ├── counterfactual.py               # What-If risk simulator
│   │   ├── explainer.py                    # Model explainability (SHAP attributions)
│   │   ├── predictor.py                    # Inference engine (raw and calibrated modes)
│   │   └── recommender.py                  # Evidence-based clinical guidelines (ACC/AHA)
│   └── database/
│       ├── db_handler.py                   # SQLite audit trail integration (audit_logs)
│       ├── models.py                       # SQLAlchemy 2.0 entities (Patient, PredictionHistory)
│       ├── session.py                      # Database engine, connection pooling & migrations
│       └── patients.db                     # SQLite database
├── backend/
│   ├── main.py                             # FastAPI application factory & lifespan
│   ├── core/
│   │   ├── config.py                       # Centralized settings (Pydantic Settings)
│   │   └── dependencies.py                 # Singleton providers & database session injection
│   ├── schemas/
│   │   ├── common.py                       # HealthResponse, APIErrorResponse
│   │   ├── patient.py                      # PatientInputSchema with clinical validation
│   │   ├── patient_management.py           # Patient CRUD & Prediction History schemas
│   │   └── prediction.py                   # Single and batch prediction schemas
│   └── api/
│       └── v1/
│           ├── api.py                      # API v1 router aggregator
│           └── endpoints/
│               ├── patients.py             # Patient management endpoints
│               └── predictions.py          # Cardiovascular risk prediction endpoints
├── artifacts/
│   ├── best_model_pipeline.joblib          # Selected best model (SVM with PCA)
│   ├── calibrated_model_pipeline.joblib    # Platt-calibrated model pipeline
│   ├── best_model_metadata.json            # Model architecture & evaluation metrics
│   ├── calibration_metrics.json            # Calibration curves & Brier score benchmarks
│   └── model_comparison.json               # CV benchmark comparison across models
├── app/
│   ├── streamlit_app.py                    # Clinical interactive web dashboard
│   └── components/
│       └── ui_widgets.py                   # Custom visual widgets, risk badges & audit trail
├── tests/                                  # Automated pytest test suites (90 tests)
│   ├── test_backend_schemas.py
│   ├── test_calibration.py
│   ├── test_counterfactual.py
│   ├── test_data_pipeline.py
│   ├── test_explainer.py
│   ├── test_model_comparison.py
│   ├── test_patient_api.py
│   ├── test_prediction_api.py
│   └── test_prediction.py
├── requirements.txt                        # Pinned dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Key Features

1. **Leak-Free Scikit-Learn Pipeline**:
   - `StandardScaler` for continuous numerical features.
   - `OneHotEncoder` for categorical factors (`Gender`).
   - `PCA(n_components=0.95)` dimensionality reduction preserving 95% of explained variance.
   - `SVC(kernel='rbf', probability=True, class_weight='balanced')` optimized via 5-fold Stratified CV.
2. **Probability Calibration & Reliability Diagrams**:
   - Platt Sigmoid calibration fitted on independent nested CV sub-splits.
   - Evaluated via Brier score, Expected Calibration Error (ECE), and Log Loss.
3. **FastAPI REST API**:
   - High-throughput single and vectorized batch inference endpoints.
   - Complete Patient Management CRUD with pagination and search.
   - Prediction history tracking linked to patient profiles.
4. **Clinical Decision Support Guidelines**:
   - Evaluates resting blood pressure ($\ge 130$ mmHg), serum cholesterol ($\ge 200$ mg/dL), fasting glucose ($\ge 126$ mg/dL), BMI ($\ge 25.0$), and resting ECG anomalies.
5. **Interactive Streamlit UI**:
   - Interactive intake form, real-time risk gauge, What-If counterfactual simulator, and historical audit trail view.

---

## 📡 REST API Endpoints

Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### System & Health
- `GET /health` or `GET /api/v1/health`: System health and artifact status check.

### Cardiovascular Risk Predictions
- `POST /api/v1/predict`: Single patient risk assessment with clinical recommendations and optional audit logging.
- `POST /api/v1/predict/batch`: High-throughput vectorized batch prediction (1 to 1000 records).

### Patient Management
- `POST /api/v1/patients`: Register a new patient profile.
- `GET /api/v1/patients`: List registered patients (supports `page`, `limit`, and `search` query parameters).
- `GET /api/v1/patients/{patient_id}`: Retrieve a patient's profile and prediction summary stats.
- `PATCH /api/v1/patients/{patient_id}`: Partially update patient demographic fields (`age`, `gender`).
- `GET /api/v1/patients/{patient_id}/predictions`: Retrieve historical risk assessments for a patient in descending timestamp order.

---

## 📋 API Usage Examples

### 1. Register a Patient
```http
POST /api/v1/patients HTTP/1.1
Content-Type: application/json

{
  "patient_id": "P-10492",
  "age": 58,
  "gender": "Male"
}
```

**Response (`201 Created`):**
```json
{
  "id": 1,
  "patient_id": "P-10492",
  "age": 58,
  "gender": "Male",
  "created_at": "2026-11-01T14:30:00Z",
  "updated_at": "2026-11-01T14:30:00Z",
  "total_predictions": 0
}
```

### 2. Predict Risk and Associate with Registered Patient
```http
POST /api/v1/predict HTTP/1.1
Content-Type: application/json

{
  "patient": {
    "Patient_ID": "P-10492",
    "Age": 58,
    "Gender": "Male",
    "Blood_Pressure": 138.0,
    "Cholesterol": 242.0,
    "BMI": 28.4,
    "Blood_Sugar": 130.0,
    "Heart_Rate": 145.0,
    "ECG_Result": 1
  },
  "calibrate": true,
  "log_audit": true,
  "require_registered_patient": true
}
```

**Response (`200 OK`):**
```json
{
  "patient_id": "P-10492",
  "predicted_class": 1,
  "risk_classification": "High Risk",
  "risk_probability": 0.6385,
  "confidence_percentage": 63.85,
  "is_calibrated": true,
  "recommendations": [
    {
      "category": "Blood Pressure",
      "status": "Stage 1/2 Hypertension Detected",
      "message": "Resting BP is elevated at 138 mmHg...",
      "level": "warning"
    }
  ],
  "audit_record_id": 14,
  "prediction_history_id": 1,
  "disclaimer": "Research prototype evaluated on synthetic data. Model estimates are statistical approximations and do not constitute clinical diagnosis."
}
```

### 3. List Patients with Pagination and Search
```http
GET /api/v1/patients?page=1&limit=20&search=10492 HTTP/1.1
```

**Response (`200 OK`):**
```json
{
  "items": [
    {
      "id": 1,
      "patient_id": "P-10492",
      "age": 58,
      "gender": "Male",
      "created_at": "2026-11-01T14:30:00Z",
      "updated_at": "2026-11-01T14:30:00Z",
      "total_predictions": 1
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 20,
  "total_pages": 1,
  "has_next": false,
  "has_prev": false
}
```

### 4. Fetch Prediction History for a Patient
```http
GET /api/v1/patients/P-10492/predictions?page=1&limit=10 HTTP/1.1
```

---

## 🛠️ Setup & Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model Pipeline
```bash
python -m src.models.train
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

### 4. Launch the FastAPI Backend Server
```bash
uvicorn backend.main:app --reload --port 8000
```
Interactive OpenAPI documentation: `http://localhost:8000/docs`

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
