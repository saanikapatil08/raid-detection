from setuptools import setup, find_packages

setup(
    name="raid-detection",
    version="0.1.0",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "datasets>=2.19.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
)
