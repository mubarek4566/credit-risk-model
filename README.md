# credit-risk-model
This project is designed for Banks in partnership with an eCommerce platform to enable Buy-Now-Pay-Later (BNPL) services through an ML-powered credit scoring system.


# 📘 Credit Scoring Business Understanding
## 1. Influence of Basel II Accord on Model Interpretability
The Basel II Capital Accord emphasizes rigorous quantitative risk measurement and regulatory compliance, which directly impacts how credit risk models must be built and documented. Financial institutions are required to demonstrate transparency, consistency, and reasoning behind their internal models, especially when calculating capital requirements for credit risk. Therefore, our credit scoring model must be not only accurate but also interpretable, traceable, and well-documented. Regulators expect institutions to explain why a customer was classified as risky, making model explainability and auditability critical for gaining internal and external trust.

## 2. Need for a Proxy Variable for Default
Since the eCommerce platform data does not include an explicit "default" label, we must construct a proxy variable to classify users as high risk (bad) or low risk (good). This proxy might be derived from behavioral signals such as Recency, Frequency, and Monetary (RFM) metrics, payment delays, or refund patterns. However, relying on a proxy introduces uncertainty and bias, especially if the proxy does not fully capture the underlying risk of true credit default. The business risks include misclassification, leading to potential financial losses, customer dissatisfaction, or regulatory scrutiny if the model unfairly denies credit or exposes the institution to high default rates.

## 3. Trade-offs Between Interpretable and Complex Models
In a regulated financial environment, there is a critical trade-off between using simple, interpretable models (e.g., Logistic Regression with Weight of Evidence [WoE]) and complex, high-performance models (e.g., Gradient Boosting Machines [GBMs]). Interpretable models are favored for regulatory transparency, auditability, and stability, allowing stakeholders to understand feature impacts and decisions. They also align well with Basel II requirements. However, these models may underperform on non-linear or high-dimensional data. On the other hand, complex models may yield better predictive accuracy but pose challenges in terms of explainability, bias detection, and regulatory approval. The optimal approach often involves balancing predictive power with regulatory interpretability, possibly using complex models for internal use and interpretable scorecards for regulatory reporting.

## Understanding datasets

| Feature Name           | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `total_amount`         | Sum of `Amount` per customer                         |
| `avg_amount`           | Mean of `Amount` per customer                        |
| `count_transactions`   | Number of transactions per customer (`Amount` count) |
| `std_amount`           | Standard deviation of `Amount`                       |
| `total_value`          | Sum of `Value` per customer                          |
| `unique_product_count` | Unique number of products per customer               |
| `unique_channel_count` | Unique number of channels used per customer          |
| `fraud_ratio`          | % of transactions that were fraud per customer       |
