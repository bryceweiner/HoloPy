from setuptools import setup, find_packages

setup(
    name="holopy",
    version="0.1.0",
    description="Holographic Universe Simulation Framework",
    author="Bryce Weiner",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "pytest-benchmark>=3.4.1",
        ],
    },
)