# Lab 01 — ML Systems

## 1. Goal

The goal of this lab was to prepare the Python environment and test two machine learning models on the Breast Cancer Wisconsin dataset.

The models used in this lab were:

* Logistic Regression
* Random Forest

I also measured their accuracy, training time, inference time, model size, and memory usage. Finally, I compared the results with the given resource budgets for Cloud, Edge, Mobile, and TinyML deployment.

## 2. Method

First, I created a Python 3.11.8 virtual environment and installed the required libraries.

The Breast Cancer Wisconsin dataset was loaded using `load_breast_cancer()`. I divided the dataset into training and testing parts using a 70/30 split. Stratification was used and `random_state=42` was set to make the results reproducible.

For the experiments I used:

* Logistic Regression with `max_iter=1000`
* Random Forest with 100 trees

The random seed was set to 42.

For the baseline test, I trained both models and calculated their accuracy on the test dataset.

For the system measurements, I used one warm-up run and then five training runs. The median training time was used as the final result.

For inference, I used one test sample, one warm-up prediction, and then 100 predictions. The median inference time was reported.

The trained models were saved using Joblib to measure their sizes. Peak memory usage was measured during training and inference.

## 3. Baseline Results

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Logistic Regression |        0.9415 |
| Random Forest       |        0.9357 |

Logistic Regression gave a slightly better accuracy than Random Forest. Its accuracy was 0.9415, while Random Forest achieved 0.9357.

## 4. System Cost Results

| Model               | Training Time (s) | Inference Time (ms) | Model Size (KB) | Peak Training Memory (MiB) | Peak Inference Memory (MiB) |
| ------------------- | ----------------: | ------------------: | --------------: | -------------------------: | --------------------------: |
| Logistic Regression |            0.4388 |              0.0573 |            1.03 |                      231.1 |                       230.6 |
| Random Forest       |            0.5254 |              6.1039 |          284.07 |                        232 |                             |
