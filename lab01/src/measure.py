import csv
import os
import statistics
import time

import joblib
import psutil
from memory_profiler import memory_usage

from common import RESULTS, get_data, make_models, set_seeds


def rss_mib():
    return psutil.Process().memory_info().rss / 1024**2


def peak_mib(func):
    res = memory_usage((func, (), {}), interval=0.01, max_usage=True)
    return res[0] if isinstance(res, (list, tuple)) else res


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = get_data()
    sample = X_test[:1]
    rows = []

    for name in make_models():
        make_models()[name].fit(X_train, y_train)  # warm-up
        times = []
        for _ in range(5):
            m = make_models()[name]
            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            times.append(time.perf_counter() - t0)
        train_s = statistics.median(times)


        model = make_models()[name]
        model.fit(X_train, y_train)


        model.predict(sample)
        lat = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(sample)
            lat.append(time.perf_counter() - t0)
        infer_ms = statistics.median(lat) * 1000

        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024

        rss_before = rss_mib()

        def train_job():
            make_models()[name].fit(X_train, y_train)

        def infer_job():
            for _ in range(100):
                model.predict(sample)

        peak_train = peak_mib(train_job)
        peak_infer = peak_mib(infer_job)

        rows.append([
            name, f"{train_s:.4f}", f"{infer_ms:.4f}", size_bytes,
            f"{size_kb:.2f}", f"{rss_before:.1f}",
            f"{peak_train:.1f}", f"{peak_infer:.1f}",
        ])
        print(rows[-1])

    header = ["model", "train_median_s", "infer_median_ms", "size_bytes",
              "size_kb", "rss_before_mib", "peak_train_mib", "peak_infer_mib"]
    with open(RESULTS / "system_cost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()