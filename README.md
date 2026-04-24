# American Express: Personalized Offer Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green.svg)

## 📌 Project Overview
This project involves building a machine learning pipeline to predict the probability of a customer engaging with specific American Express offers. The challenge lies in processing multi-source data (transactions, events, and metadata) and managing a high degree of class imbalance to deliver top-7 personalized recommendations per user.

## 📊 Data Architecture & Integration
The system integrates several large-scale datasets:
* **Transactions (`add_trans`):** Historical spending patterns across various industry categories.
* **Event Logs (`add_event`):** User interaction history, impressions, and activity timestamps.
* **Offer Metadata:** Contextual details regarding the marketing offers.
* **Train/Test Sets:** User-offer interaction labels.


## 🛠️ Feature Engineering (The Core Strength)
To maximize model performance, I engineered features across three domains:
1. **Spending Behavior:** Aggregated metrics (Sum, Mean, Max, Min) of customer spending and "Transaction Density" (transactions per active month).
2. **Temporal Patterns:** Extracted `most_common_hour` and `most_common_day` to understand when users are most receptive to offers.
3. **Engagement Lifecycle:** Calculated `event_span_days` and `days_since_last_event` to measure user recency and loyalty.
4. **Data Density Flags:** Created indicators (`has_trans_data`, `has_event_data`) to help the model distinguish between "cold-start" users and high-activity users.

## 🤖 Modeling & Training Strategy
The project utilizes **LightGBM** due to its efficiency with sparse data and its ability to handle categorical features natively.

### 1. Handling Class Imbalance
The dataset is highly skewed (approx. 95% negative). I addressed this by:
* **Scale Pos Weight:** Setting `scale_pos_weight: 20` to penalize the model more for missing positive engagements.
* **Stratified K-Fold:** Implementing 5-Fold Stratified Cross-Validation to ensure each fold maintains the same percentage of positive samples.

### 2. Hyperparameters
* **Boosting Type:** GBDT (Gradient Boosting Decision Tree)
* **Learning Rate:** 0.02
* **Max Depth:** 7 (to prevent overfitting on noisy event data)
* **Regularization:** Applied L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties.


## 📈 Performance & Evaluation
* **Primary Metric:** ROC-AUC
* **CV Result:** Achieved a stable average AUC of **~0.6117** across folds.
* **Output:** The model ranks potential offers for each user, providing the **Top 7** most likely conversions for deployment.

## 📂 Repository Structure
* `amex-r2-final.ipynb`: The complete end-to-end pipeline (Data loading to Submission).
* `submission.csv`: Final predictions formatted for evaluation.
* `requirements.txt`: List of dependencies.

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the notebook `amex-r2-final.ipynb` in a Kaggle or Jupyter environment.
