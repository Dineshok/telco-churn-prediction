# 📡 Telco Customer Churn Prediction

A machine learning project to predict customer churn using the IBM Telco dataset (7,043 customers, 21 features).

The primary objective is to **identify at-risk customers early**, prioritizing **recall and F1-score over raw accuracy**, since missing a churner is more costly than a false alarm.

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

* **Dataset:** IBM Telco Customer Churn
* **Target:** `Churn` (1 = churn, 0 = stay)
* **Class imbalance:** ~73% non-churn / ~27% churn

👉 This imbalance makes **accuracy misleading**, so evaluation focuses on:

* Precision
* Recall
* F1-score (for churn class)

---

## 📊 Key EDA Insights

1. **Tenure is the strongest signal**
   → Most churn happens within the first 12 months

2. **Contract type matters heavily**
   → Month-to-month customers churn the most

3. **Fiber optic customers churn more**
   → Despite being a premium service

---

## ⚙️ Preprocessing Pipeline

* Dropped `customerID` (non-informative)
* Fixed `TotalCharges` (string → numeric, handled missing)
* Encoded target (`Yes → 1`, `No → 0`)
* One-hot encoding with `drop_first=True`
* Standardized numerical features using `StandardScaler`
* Train-test split (80/20) with `stratify=y`

---

## 🤖 Model Training & Threshold Tuning

Instead of relying on default predictions, **decision thresholds were tuned** for each model to maximize F1-score.

### 🔥 Best Threshold Results

| Model               | Threshold | Precision | Recall    | F1-Score  |
| ------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression | **0.566** | 0.541     | **0.743** | 0.626     |
| Decision Tree       | 0.606     | 0.592     | 0.660     | 0.625     |
| AdaBoost            | 0.475     | **0.592** | 0.668     | **0.628** |
| XGBoost             | 0.525     | 0.544     | 0.741     | 0.627     |

---

## 🏆 Final Model Selection

**Logistic Regression was selected** as the final model.

### Why?

* High **recall (0.743)** → captures most churners
* Strong **F1-score (0.626)**
* More interpretable than ensemble models
* Stable performance across thresholds

👉 Even though AdaBoost had slightly higher F1, Logistic Regression was preferred for:

* Interpretability (important for business use)
* Consistency with SHAP explanations

---

## 🎯 Key Insight: Threshold Optimization

Instead of using the default **0.5 threshold**, the model uses:

👉 **Optimal threshold = 0.566**

This improves:

* Recall ↑ (detect more churners)
* Overall F1-score ↑

---

## 🔎 Model Explainability (SHAP)

SHAP was used to understand **why the model predicts churn**.

### Global Feature Importance (Top Drivers)

* `tenure` → strongest negative correlation with churn
* `InternetService_Fiber optic` → increases churn risk
* `Contract_Two year` → strongly reduces churn
* `MonthlyCharges` → higher charges increase risk
* `TotalCharges` → correlated with churn patterns

---

### Local Explanation (Waterfall Plot)

For individual customers:

* Each feature pushes prediction **towards or away from churn**
* Final prediction = sum of all feature contributions

---

## 🚀 Deployment

The model is deployed as a **Streamlit web app**:

👉 [Live Demo](https://huggingface.co/spaces/Dineshok/churn-predictor)

### Features:

* User-friendly input form
* Real-time churn probability
* Threshold-based prediction
* Risk factor breakdown

---

## 🛠 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Key Takeaways

* Accuracy is misleading for imbalanced data
* Threshold tuning significantly improves real-world performance
* Simpler models (Logistic Regression) can outperform complex ones when properly tuned
* Explainability (SHAP) is crucial for business trust

---
