from setuptools import setup
import os
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

import re
VERSIONFILE="coolpuppy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['Cython', 'numpy', 'cooler', 'pandas', 'natsort',
                        'scipy', 'cooltools', 'pyyaml']

setup(
      name='coolpuppy',
      version=verstr,
      packages=['coolpuppy'],
      entry_points={
          'console_scripts': ['coolpup.py = coolpuppy.__main__:main',
                              'plotpup.py = coolpuppy.__plotpuppy_main__:main']},
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3.6',
      description='A versatile tool to perform pile-up analysis on Hi-C data in .cool format.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      project_urls={'Source':'https://github.com/Phlya/coolpuppy',
                    'Issues':'https://github.com/Phlya/coolpuppy/issues'},
      author='Ilya Flyamer',
      author_email='flyamer@gmail.com',
      classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    zip_safe=False
)
