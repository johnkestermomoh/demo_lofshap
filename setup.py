from setuptools import setup, find_packages
from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
requirements = ["requests>=2"]

setup(
    name="demo_lofshap",
    version="0.1.0",
    description="Local Outlier Factor combining shapley values for detecting outlier and labelling signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnkestermomoh/demo_lofshap.git",
    packages=[find_packages(), 'demo_lofshap'],
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)