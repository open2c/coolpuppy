from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
      name='coolpuppy',
      version='0.8.5',
      packages=['coolpuppy'],
      entry_points={
          'console_scripts': ['coolpup.py = coolpuppy.__main__:main',
                              'plotpup.py = coolpuppy.__main__:plotpuppy']},
      install_requires=['numpy', 'cooler', 'pandas', 'natsort', 'scipy',
                        'cooltools'],
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
    ]
)
