from setuptools import setup
import os
from os import path
import io

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


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != "" and not (req.startswith("#") or req.startswith("-"))
    ]


setup_requires = [
    "cython",
    "numpy",
]

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = get_requirements("requirements.txt")

setup(
    name="coolpuppy",
    version=verstr,
    packages=["coolpuppy", "coolpuppy.lib"],
    entry_points={
        "console_scripts": [
            "coolpup.py = coolpuppy.CLI:main",
            "plotpup.py = coolpuppy.plotpuppy_CLI:main",
            "dividepups.py = coolpuppy.divide_pups_CLI:main",
        ]
    },
    setup_requires=setup_requires,
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.8",
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
