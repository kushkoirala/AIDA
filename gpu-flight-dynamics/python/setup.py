#!/usr/bin/env python
"""
Setup script for GPU-Accelerated Flight Dynamics Simulator
"""

from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gpu-flight-dynamics",
    version="1.0.0",
    author="Kushal Koirala",
    author_email="kush.koirala@gmail.com",
    description="GPU-Accelerated 6-DOF Flight Dynamics Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kushkoirala/gpu-flight-dynamics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x",
            "torch>=2.0",
        ],
        "rl": [
            "gymnasium",
            "stable-baselines3",
        ],
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    keywords=[
        "flight dynamics",
        "simulation",
        "CUDA",
        "GPU",
        "aerospace",
        "reinforcement learning",
        "6-DOF",
    ],
)
