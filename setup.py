#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="hyperagent",
    version="0.0.1",
    python_requires=">=3.7",
    packages=find_packages(exclude=["scripts", "experiments", "atari_results"]),
    install_requires=[
        "gym",
        "tqdm",
        "numpy>1.16.0",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard>=2.5.0",
        "torch>=1.8.1",
        "numba>=0.51.0",
        "h5py>=2.10.0",  # to match tensorflow's minimal requirements
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "sphinx<4",
            "sphinx_rtd_theme",
            "sphinxcontrib-bibtex",
            "flake8",
            "flake8-bugbear",
            "yapf",
            "isort",
            "pytest",
            "pytest-cov",
            "ray>=1.0.0,<1.7.0",
            "wandb>=0.12.0",
            "networkx",
            "mypy",
            "pydocstyle",
            "doc8",
            "scipy",
        ],
        "atari": ["atari_py", "opencv-python"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    },
)
