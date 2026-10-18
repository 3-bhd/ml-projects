# ML Projects

Omar Abdelhady

Two end-to-end machine learning projects, each with a full notebook, write-up, figures and results.

## [Credit Default Prediction](credit-default-prediction/)

Predicts whether a credit card client will default next month from six months of billing and repayment
history, and picks the decision threshold from an explicit cost ratio instead of accuracy. Random Forest,
test ROC-AUC 0.782, catches 72% of defaulters at the chosen threshold (0.17). Dataset: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (30,000 clients).

## [Heart Disease Prediction](heart-disease-prediction/)

Predicts heart disease from routine clinical data, tuned to prioritize recall since missing a sick patient
is the costlier mistake. Logistic Regression, nested-CV ROC-AUC 0.892, test ROC-AUC 0.958, threshold tuned
for 90% recall. Dataset: [UCI Heart Disease, Cleveland subset](https://archive.ics.uci.edu/dataset/45/heart+disease) (303 patients).

## Tech stack

Python, pandas, numpy, scikit-learn, matplotlib; XGBoost and SHAP for the heart disease project. Each
subfolder has its own pinned `requirements.txt` — set up a separate environment per project.

**Educational models only — not for real credit or medical decisions.**
