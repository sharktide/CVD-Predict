from setuptools import setup, find_packages

setup(
    name="ohca-predictor",
    version="1.0.0",
    description="Multimodal Wearable AI for Out-of-Hospital Cardiac Arrest Prediction",
    author="OHCA Predictor Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "requests>=2.28.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.12.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ohca-train=ohca_predictor.scripts.train:main",
            "ohca-evaluate=ohca_predictor.scripts.evaluate:main",
            "ohca-generate=ohca_predictor.scripts.generate_data:main",
            "ohca-profile=ohca_predictor.scripts.profile:main",
            "ohca-benchmark=ohca_predictor.scripts.benchmark:main",
        ],
    },
)
