from setuptools import setup, find_packages

setup(
    name="smog_hazard_index",
    version="1.0.0",
    description="Machine Learning for Ground-Level PM2.5 Estimation & Development of Smog Hazard Index (SHI)",
    author="EHDS Lab / SUPARCO Internship Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "pyyaml",
        "openpyxl",
        "scikit-learn",
        "xgboost",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "shapely",
    ],
)
