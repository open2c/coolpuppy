from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
      name='coolpuppy',
      version='0.8.2',
      scripts=['coolpup.py'],
      install_requires=['numpy', 'cooler', 'pandas', 'natsort', 'scipy',
                        'cooltools'],
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
    ]
)
