from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bayesian-rlayernorm",
    version="1.0.0",
    author="Mohsen Mostafa",
    author_email="mohsen.mostafa.ai@outlook.com",
    description="Bayesian R-LayerNorm: Uncertainty-Aware Robust Normalization for Noisy Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bayesian-r-layernorm",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/bayesian-r-layernorm/issues",
        "Documentation": "https://bayesian-rlayernorm.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/bayesian-r-layernorm",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "nbsphinx>=0.8.0",
        ],
        "cloud": [
            "colab-ssh>=0.3.0",
            "gdown>=4.6.0",
            "kaggle>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bayesian-train=bayesian_rlayernorm.cli.train:main",
            "bayesian-eval=bayesian_rlayernorm.cli.evaluate:main",
            "bayesian-visualize=bayesian_rlayernorm.cli.visualize:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bayesian_rlayernorm": [
            "configs/*.yaml",
            "data/*.json",
        ],
    },
    license="MIT",
    keywords="deep-learning, normalization, robustness, uncertainty, bayesian, pytorch",
)
