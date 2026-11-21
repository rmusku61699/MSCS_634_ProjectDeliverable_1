# MSCS 634 – Project Deliverable 1: Data Collection, Cleaning, and Exploration

**Student:** Ruchitha Musku
**Course project:** Advanced Data Mining for Data-Driven Insights and Predictive Modeling
**Dataset:** IBM Telco Customer Churn

---

## Repository contents

| File / folder | Description |
|---|---|
| `MSCS_634_ProjectDeliverable_1.ipynb` | Main notebook: loading, cleaning, EDA and insights, with commented code and outputs |
| `data/Telco-Customer-Churn.csv` | Raw dataset (7,043 rows × 21 columns) |
| `data/telco_churn_cleaned.csv` | Cleaned dataset produced by the notebook (input for Deliverables 2–4) |
| `figures/` | All visualisations exported as PNG files |
| `requirements.txt` | Python packages needed to run the notebook |

**To run:** `pip install -r requirements.txt`, then open the notebook in Jupyter and choose *Run All*. All paths are relative to the repository folder.

---

## Dataset summary

The **IBM Telco Customer Churn** dataset (public IBM sample data, also on Kaggle as `blastchar/telco-customer-churn`) describes **7,043 customers** of a fictional California telecom company. It has **21 attributes**:

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account:** tenure (months), Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Target:** Churn (customer left within the last month: Yes/No)

**Why this dataset:** it is well above the size requirement (7,043 records, 20 usable features), mixes numeric and categorical data, has real data-quality problems to fix, and supports every later deliverable: regression (MonthlyCharges/TotalCharges), classification (Churn), clustering (customer segments) and association rules (service bundles). Customer retention is also a high-value business problem.

---

## Data cleaning steps

| # | Issue found | Action and reason |
|---|---|---|
| 1 | `TotalCharges` loaded as text; **11 blank values** hidden from `isnull()` | Converted to numeric. All 11 rows had `tenure = 0` (new, not-yet-billed customers), so the values were **imputed as $0**, the logically correct total, instead of the mean ($2,280) or dropping valid customers. |
| 2 | `SeniorCitizen` coded as 0/1 while all other flags use Yes/No | Mapped to No/Yes for consistency. |
| 3 | "No internet service" / "No phone service" levels in 7 columns duplicate the information in `InternetService` / `PhoneService` | Verified they match exactly, then merged them into "No" to avoid redundant categories and collinear dummy variables. |
| 4 | Possible duplicates | 0 exact duplicate rows and 0 repeated customer IDs. 22 rows match on every attribute except ID; these are different customers with common profiles, so they were **kept**. |
| 5 | Noisy data | Range checks (no negatives or impossible values), a cross-field check (`TotalCharges` vs `tenure × MonthlyCharges`: median ratio 1.00, 98% between 0.86 and 1.16) and IQR outlier detection (**0 outliers**) all passed. The skew in `TotalCharges` reflects real long-tenure customers, so the values were kept. |
| 6 | `customerID` has no analytic value | Dropped from the analysis set after the duplicate check. A `Churn_flag` (1/0) column was added, and the cleaned data was saved to `data/telco_churn_cleaned.csv`. |

---

## Exploration performed

- Churn class balance (bar chart)
- Distributions of tenure, MonthlyCharges and TotalCharges by churn (histograms with KDE, skewness)
- Box plots for outlier detection and for comparing numeric features by churn
- Churn rate for every categorical feature (16-panel bar grid)
- Tenure vs MonthlyCharges scatter plot and churn by tenure band
- Full correlation heatmap, correlation of each feature with churn, and a multicollinearity check
- Contract × InternetService churn heatmap and churn by number of add-on services

---

## Key insights

1. **Class imbalance:** 26.5% of customers churned (1,869 of 7,043).
2. **Contract type is the strongest driver:** month-to-month customers churn at **42.7%**, versus 11.3% (one-year) and 2.8% (two-year).
3. **Churn happens early:** **52.9%** churn in the first 6 months, falling to 6.6% after 5 years. Median tenure is 10 months for churners and 38 for customers who stayed.
4. **Fiber-optic customers churn at 41.9%** (DSL 19.0%). Fiber customers on month-to-month contracts are the highest-risk group at **54.6%**.
5. **Electronic-check payers churn at 45.3%**, versus 15–17% for automatic payment methods.
6. **Support add-ons go with retention:** customers without OnlineSecurity or TechSupport churn at about 31%, versus about 15% for those who have them. Among internet customers, churn falls from 46% (1 add-on) to 5% (all 6 add-ons).
7. **Demographics:** seniors churn at 41.7% (vs 23.6%), and customers without a partner or dependents churn more. **Gender and PhoneService have essentially no effect.**
8. **Multicollinearity:** `TotalCharges` correlates strongly with `tenure` (r = 0.83) and `MonthlyCharges` (r = 0.65). `MonthlyCharges` correlates strongly with fiber service (r = 0.79).

### How the insights guide the next deliverables
- **Deliverable 2 (Regression):** predict `MonthlyCharges` from the service mix; use Ridge/Lasso because of the correlated predictors; engineer `num_addons`, tenure bands and an `auto_payment` flag.
- **Deliverable 3 (Classification):** predict `Churn` using a stratified split and F1 / ROC-AUC because of the imbalance; scale numeric features for k-NN/SVM.
- **Deliverable 3 (Clustering & association rules):** segment customers on tenure, charges and services; mine rules such as {Month-to-month, Fiber optic, No TechSupport} → {Churn = Yes}.
- Low-signal features (gender, PhoneService) are candidates for removal during feature selection.

---

## Challenges and how they were addressed

| Challenge | Solution |
|---|---|
| Missing values hidden as blank strings, so `isnull()` reported a complete dataset | Checked for whitespace-only strings and the unexpected text dtype, then converted with `pd.to_numeric(errors="coerce")` to expose them. |
| Choosing an imputation method | Looked at the affected rows instead of applying a blanket mean/median. The shared `tenure = 0` pattern showed the values were structurally missing, which supported imputing $0. |
| Deciding whether same-profile rows were duplicates | Separated exact/ID duplicates (none) from attribute-level matches, and kept the latter as legitimate distinct customers. |
| Telling noise from genuine skew in `TotalCharges` | Combined the IQR rule with a domain consistency check against `tenure × MonthlyCharges`. Both confirmed that the long tail is real, so the skew will be handled with transformation in modelling rather than by removing data. |
| Redundant category levels inflating dimensionality | Verified with assertions that "No internet/phone service" exactly mirrors the parent column before merging, so no information was lost. |
| Mixed data types in one correlation analysis | Binary-encoded Yes/No columns, ordinal-encoded Contract and one-hot encoded multi-level columns before computing correlations. |

---

## References
- IBM. *Telco Customer Churn* sample dataset. https://github.com/IBM/telco-customer-churn-on-icp4d
- Kaggle mirror: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- McKinney, W. *pandas* documentation. https://pandas.pydata.org/docs/
- Waskom, M. *seaborn: statistical data visualization*. https://seaborn.pydata.org/
