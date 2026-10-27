# Data Card: UCI Adult (processed)

- **Source:** UCI Machine Learning Repository, Adult (Census Income), loaded via
  `fetch_openml(name='adult', version=2, as_frame=True)`. Extracted from the 1994 US Census database.
- **License:** CC BY 4.0 (as listed by UCI; verify on the dataset page).
- **Size:** raw 48,842 rows x 15 columns. Stratified 70/30 split (seed 42): train 34,189,
  test 14,653. After removing 24 exact duplicates from train: train 34,165.
  Processed file: 48,818 rows x 18 columns (14 features + income + capital_net,
  hours_per_age + split).
- **Processed data SHA256:** 79f57ec7c943ca09cb5171a0d04253c82fed7b37ec85c10aa185aa615c8b720f
- **Pipeline:** stratified split, train-only median/mode imputation, 1st/99th percentile
  clipping from train, train-only exact-duplicate removal, derived features
  capital_net and hours_per_age. Fitted transformer: results/preprocessor.joblib.
- **Validation results:** 0 type violations, 0 range violations. Missing values:
  workclass 2,799; occupation 2,809; native-country 857. Exact duplicate rows in raw data: 52.
- **Label quality:** 5 groups of duplicate feature rows (10 rows) with conflicting income
  labels. Manual check of 50 random rows: 0 impossible values.
- **Known biases:** 1994 US data (outdated); the $50K threshold is not inflation-adjusted;
  sex and race imbalance (majority male, majority White); most records have
  native-country = United-States; missing values concentrated in workclass, occupation
  and native-country; `fnlwgt` is a census sampling weight.
- **Privacy concerns:** census-derived and anonymized, but quasi-identifiers (age, sex, race,
  native-country, marital-status) can enable re-identification when linked with other data;
  sensitive attributes (race, sex) must not be used for discriminatory decisions.
- **Reproduce:** `pip install -r requirements.txt && python lab04/src/run_pipeline.py`