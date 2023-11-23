#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="pie",
    version="1.0.0",
    author="IDRIS",
    author_email="assist@idris.fr",
    url="https://www.idris.fr",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["pie = pie:cli"],
    },
    install_requires=[
        "deepspeed",
        "idr_torch>=2.0.0",
        "mlflow",
        "torch",
        "torchmetrics",
        "tqdm",
        "transformers",
    ],
    license="MIT",
)
