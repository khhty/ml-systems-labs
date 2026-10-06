import csv
from sklearn.metrics import accuracy_score
from common import RESULTS, get_data, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = get_data()
    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append((name, f"{acc:.4f}"))
        print(name, f"{acc:.4f}")

    with open(RESULTS / "baseline_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_accuracy"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()