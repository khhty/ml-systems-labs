import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm",
]


def main():
    lines = [f"Python {platform.python_version()}", f"Platform {platform.platform()}"]
    for pkg in PACKAGES:
        try:
            lines.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            lines.append(f"{pkg}: NOT INSTALLED")
    text = "\n".join(lines)
    print(text)
    out = Path(__file__).resolve().parent.parent / "results" / "versions.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()