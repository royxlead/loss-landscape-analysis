from setuptools import find_packages, setup

setup(
    name="loss-fn-analysis",
    version="1.0.0",
    description="Research project analyzing gradient behavior of MSE vs Cross-Entropy on MNIST",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
