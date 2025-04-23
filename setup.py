from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = "LogitLens",
    version = "0.1",
    description = "API for logit lens plots",
    author = "Gwenyth Lincroft",
    author_email = "lincroft.g@northeastern.edu",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='=3.11'
)