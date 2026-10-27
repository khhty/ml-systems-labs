# Lab 4: Data pipeline and quality

## 1. Problem definition

In this lab I work with the UCI Adult dataset, loaded with
`fetch_openml(name='adult', version=2, as_frame=True)`. It has 48,842 rows.

**Task.** Predict if a person earns more than $50K per year. This is a binary
classification problem.

**Inputs.** 14 features: age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week, native-country.

**Output.** income >50K (positive class) or <=50K.

**Success metrics.** I use F1 and accuracy on a 30% test set. I use F1 together with
accuracy because the classes are not balanced (only about 24% of people earn >50K),
so accuracy alone can look better than the model really is.

**Constraints.**
1. No leakage. I split the data first and clean it after. All statistics (median,
   mode, percentiles) are learned from the train set only. The test set is not changed.
2. Reproducible pipeline. I use a fixed seed, a YAML config, one command to run
   everything, and a SHA256 hash of the final dataset.

**Risks.** Missing values, duplicate rows with different labels, and bias in
sex, race and native-country.

## 2. Ingestion

I load the raw data and do not change it. The only thing I do is rename the target
column `class` to `income`. The log shows 48842 rows and 15 columns, as expected.

## 3. Validation

I checked the data against the schema from the task. No rows were dropped.

| Check | Result |
|---|---|
| Type violations | 0 |
| Range violations (age, hours-per-week, capital-gain, capital-loss) | 0 / 0 / 0 / 0 |
| Missing values | workclass 2799, occupation 2809, native-country 857 |
| Exact duplicate rows (whole dataset) | 52 |
| Income values | only `<=50K` and `>50K` |

Missing values are allowed by the null policy, so I only report them. The full
report is in `results/validation_report.json`.

## 4. Cleaning and transformation

- I split the data 70/30 with stratification before cleaning (seed 42).
  Train = 34189 rows, test = 14653 rows.
- I fit the imputer on train only: median for numeric columns, mode for categorical.
- I clip numeric columns to the 1st and 99th percentile taken from train.
- I drop exact duplicates from train only: 24 rows removed, so train = 34165.
  The test set stays untouched.
- The fitted transformer is saved with `joblib.dump` to `results/preprocessor.joblib`.

## 5. Feature engineering

I added two new features:
- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)` (the denominator is at least 1,
  to avoid division by zero)

I tested them with a LogisticRegression baseline:

| Variant | Accuracy | F1 |
|---|---|---|
| Baseline (without new features) | 0.8540 | 0.6601 |
| With capital_net and hours_per_age | 0.8578 | 0.6696 |

Both metrics got a bit better: accuracy +0.0038 and F1 +0.0095. The improvement is
small, but it is consistent. The new features were made from already cleaned
columns, so no information from the test set was used.

## 6. Label quality

- **Duplicate-conflict audit.** I found 5 groups of rows with identical features but
  different income labels (10 rows in total). This is label noise, and no model can
  predict both labels for the same input.
- `education` and `education-num` match each other in the whole dataset.
- **Manual check of 50 random rows** (`results/sample_50_rows.csv`): I found 0
  impossible values. Ages are 18-67, hours are 4-84, max capital-gain is 14344 and
  max capital-loss is 3004, so everything is inside the allowed ranges. Education
  matches education-num in every row, all categories are from the normal lists, and
  relationship matches sex (Husband = Male, Wife = Female). 3 rows have missing
  values, which is allowed. One value looks strange but is valid: row 3375 has
  hours-per-week = 4.

## 7. Reproducibility

The whole pipeline runs with one command:

`python lab04/src/run_pipeline.py`

I ran it two times. The SHA256 hash of the processed dataset was the same both times:

`79f57ec7c943ca09cb5171a0d04253c82fed7b37ec85c10aa185aa615c8b720f`

![Run 1](results/screenshots/run1_hash.jpg)
![Run 2](results/screenshots/run2_hash.jpg)

Config: `config.yaml`. Data card: `data_card.md`.

## Conclusion

I built a pipeline that is leakage-free and can be reproduced. The data has no type
or range errors, but it has missing values, a few duplicates and a few conflicting
labels. The two new features gave a small improvement for the baseline model.