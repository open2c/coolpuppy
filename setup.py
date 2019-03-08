from setuptools import setup

setup(
      name='coolpuppy',
      version='0.7.3',
      scripts=['coolpup.py'],
      install_requires=['numpy', 'cooler', 'pandas', 'natsort', 'scipy',
                        'mirnylib', 'cooltools']
)
