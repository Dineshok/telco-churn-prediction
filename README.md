# 📡 Telco Customer Churn Prediction

A machine learning project to predict customer churn using the IBM Telco Customer Churn dataset (7,043 customers, 21 features). The goal was to identify at-risk customers early — minimizing missed churners matters more than avoiding false alarms, which shaped every modelling decision.

---

## 📁 Project Structure

```
├── telco_churn_analysis.ipynb   # Full pipeline: EDA → preprocessing → modelling → explainability
├── app.py                       # Streamlit prediction app
├── requirements.txt
└── README.md
```

---

## 🔍 Problem & Dataset

- **Dataset:** IBM Telco Customer Churn (via GitHub, no login required)
- **Target:** `Churn` — whether a customer left within the last month
- **Class imbalance:** 73.5% No Churn / 26.5% Churn → prioritised recall over raw accuracy

---

## 📊 Key EDA Findings

Three patterns stood out before any modelling:

1. **Tenure vs Churn** — churn is heavily concentrated in the first 12 months; long-tenure customers almost never leave
2. **Contract type** — month-to-month customers churn at ~43%; one-year and two-year customers barely churn at all
3. **Fiber optic internet** — disproportionately high churn segment despite being a premium service

---

## ⚙️ Preprocessing

| Step | Decision | Reason |
|---|---|---|
| Drop `customerID` | Not predictive | Just an identifier |
| Fix `TotalCharges` | Stored as string; 11 blank rows (tenure=0) filled with 0 | Avoids silent NaN issues |
| Encode target | `Yes → 1`, `No → 0` | Binary classification |
| One-hot encode | `drop_first=True` on all categorical columns | Avoids multicollinearity |
| Scale numerics | `StandardScaler` fit on train only | Prevents data leakage into test set |
| Train/test split | 80/20 with `stratify=y` | Preserves 26.5% churn ratio in both splits |

---

## 🤖 Model Comparison

Four classifiers were trained and evaluated on the **churn class** (not overall accuracy — a model that predicts "No" for everyone gets 73.5% accuracy and is useless):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.660 | **0.561** | **0.607** |
| Decision Tree | 0.725 | 0.482 | 0.479 | 0.481 |
| AdaBoost | 0.804 | 0.668 | 0.521 | 0.586 |
| XGBoost | 0.801 | 0.656 | 0.524 | 0.582 |

**Logistic Regression was selected** — it achieved the highest F1-score (0.607) and recall (0.561) on the churn class, despite XGBoost being the more complex model. For a churn use case, recall matters most: a missed churner costs more than a false alarm.

**Threshold tuning:** The default 0.5 threshold was lowered to **0.4** to further improve recall, accepting a small drop in precision. This is a deliberate business trade-off — it's cheaper to proactively retain an extra customer than to lose one silently.

---

## 🔎 SHAP Explainability

SHAP `LinearExplainer` was applied to the final model to understand *why* it makes predictions, not just what it predicts.

**Global feature importance (top 5):**
1. `tenure` — by far the strongest signal; longer-tenured customers are much less likely to churn
2. `InternetService_Fiber optic` — positively associated with churn
3. `Contract_Two year` — strongly protective against churn
4. `TotalCharges` / `MonthlyCharges` — higher charges correlate with higher churn risk
5. `Contract_One year` — moderately protective

**Single-customer waterfall plot** confirmed the model's logic on an individual churned customer: low tenure pushed the prediction toward churn (+0.82), while not having fiber optic pulled it back (−0.59).

---

## 🚀 Live Demo

Deployed as a Streamlit app on Hugging Face Spaces — enter customer details and get an instant churn probability with a risk factor breakdown.

👉 [Try the app](https://huggingface.co/spaces/Dineshok/churn-predictor) 

---

## 🛠 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 Potential Improvements

- Try Random Forest or stacked ensemble for better recall
- Address class imbalance directly with SMOTE or class weights
- Add SHAP explanations per prediction in the Streamlit app
- Collect more features: customer support call history, usage patterns
