from setuptools import setup
import os
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

import re

VERSIONFILE = "coolpuppy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = [
        "Cython",
        "cooler",
        "numpy>=1.16.5",
        "scipy",
        "cooltools>=0.5.0,<=5.1",
        "pyyaml",
        "more_itertools",
        "seaborn",
        "natsort",
        "tables",
        "h5sparse",
        "multiprocessing_logging",
    ]

setup(
    name="coolpuppy",
    version=verstr,
    packages=["coolpuppy"],
    entry_points={
        "console_scripts": [
            "coolpup.py = coolpuppy.CLI:main",
            "plotpup.py = coolpuppy.plotpuppy_CLI:main",
            "dividepups.py = coolpuppy.divide_pups_CLI:main",
        ]
    },
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.6",
    description="A versatile tool to perform pile-up analysis on Hi-C data in .cool format.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/open2c/coolpuppy",
        "Issues": "https://github.com/open2c/coolpuppy/issues",
    },
    author="Open2C",
    author_email="flyamer@gmail.com",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
