from setuptools import setup, find_packages

setup(
    name="skywalker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "requests>=2.25.0",
        "pyarrow>=3.0.0",  # For parquet support
    ],
    author="YourName",
    author_email="your.email@example.com",
    description="A package for retrieving and saving news articles from the GDELT API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/skywalker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

