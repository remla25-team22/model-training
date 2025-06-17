# model-training/setup.py
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="model-training",       
    version="0.0.1",
    python_requires=">=3.9",

    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
