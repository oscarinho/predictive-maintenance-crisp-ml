# 🔧 Predictive Maintenance - CRISP-ML Project

A complete **CRISP-ML** (Cross-Industry Standard Process for Machine Learning) pipeline for predicting machine failures.

## 📁 Project Structure

```
├── app/
│ └── app.py # Streamlit deployment app
│
├── data/
│ └── predictive_maintenance.csv # Dataset
│
├── model/
│ ├── model.pkl # Trained pipeline model
│ ├── label_encoder_type.pkl # Encoder for product Type
│ ├── feature_cols.pkl # Feature names (model schema)
│ └── model_metadata.pkl # Metrics, params, metadata
│
├── notebooks/
│ └── CRISP_ML_Predictive_Maintenance.ipynb # Phases 1–5
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

Dataset from Kaggle:  
🔗 *Machine Predictive Maintenance Classification*

### Input Features

| Feature | Description |
|------|-------------|
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Machine operating temperature |
| Rotational speed [rpm] | Rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool usage |
| Type | Product quality (L, M, H) |

**Target**
- `0` → No Failure  
- `1` → Failure  

---

## 🔄 CRISP-ML Pipeline

### Phase 1 — Business Understanding
- **Goal:** Predict failures before they occur
- **Business Value:** Reduce downtime, optimize maintenance
- **Primary KPI:** Recall (minimize missed failures)

---

### Phase 2 — Data Understanding
- 10,000 observations
- Strong class imbalance (~3.4% failures)
- Multiple failure mechanisms
- No missing values

---

### Phase 3 — Data Preparation

**Dropped Columns**
- UDI
- Product ID

**Encoding**
- `Type` → numerical encoding (L, M, H)

**Feature Engineering**
- `Temp_diff` → Process − Air temperature
- `Power` → Torque × Rotational speed
- `Strain` → Torque × Tool wear
- `Tool_wear_ratio` → Normalized tool wear
- `Temp_ratio` → Process / Air temperature

**Splitting**
- Stratified train/test split (80/20)

**Preprocessing**
- All preprocessing handled **inside the model pipeline**
- No standalone scaler artifact

---

### Phase 4 — Modeling

Models evaluated:
- Logistic Regression (baseline)
- Random Forest (balanced)
- Gradient Boosting
- LightGBM (optimized, low-noise)

**Evaluation Strategy**
- Stratified cross-validation
- F1-score as primary selection metric
- Threshold optimization for business trade-offs

---

### Phase 5 — Evaluation

Final model evaluated on hold-out test set.

| Metric | Value |
|------|------|
| Accuracy | ~99% |
| Precision | >90% |
| Recall | >80% |
| F1 Score | ~0.88 |
| ROC-AUC | >0.95 |

**Key Insights**
- Failures are driven by combined stress, not single variables
- Torque × speed interaction is critical
- Tool wear and power are dominant predictors

---

### Phase 6 — Deployment

Deployment via **Streamlit**.

**Capabilities**
- Real-time predictions
- Probability-based decision support
- Business-friendly UI
- Pipeline-safe inference

---


*Built with ❤️ using CRISP-ML methodology and Streamlit*
# predictive-maintenance-crisp-ml
# predictive-maintenance-crisp-ml
