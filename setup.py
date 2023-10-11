try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

from distutils.command.install import install
import os
import json

here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, "README.md")) as f:
        README = f.read()
except UnicodeDecodeError:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        README = f.read()

setup(
    name="tfmesos2",
    version="0.1.0",
    description="Tensorflow for Apache Mesos",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=["tensorflow>=2.0.0", "avmesos>=0.4.0", "addict>=2.4.0", "flask>=3.0.0", "flask_httpauth>=4.0.0", "waitress>=2.1.0"],
    setup_requires=["tensorflow>=2.0.0", "avmesos>=0.4.0", "addict>=2.4.0", "flask>=3.0.0", "flask_httpauth>=4.0.0"],
    author="Andreas Peters",
    author_email="support@aventer.biz",
    url="https://www.aventer.biz/",
    python_requires=">=3.6",
)
