from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
      name='coolpuppy',
      version='0.7.6',
      scripts=['coolpup.py'],
      install_requires=['numpy', 'cooler', 'pandas', 'natsort', 'scipy',
                        'cooltools'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      project_urls={'Source':'https://github.com/Phlya/coolpuppy',
                    'Issues':'https://github.com/Phlya/coolpuppy/issues'}
)
