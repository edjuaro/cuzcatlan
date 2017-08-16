from setuptools import setup

setup(name='cuscatlan',
      version='0.0.3',
      description="Edwin Juarez's personal library. Version 0.0.3 adds one proof-of-concept function (custom_pearson).",
      url='https://github.com/edjuaro/cuscatlan',
      author='Edwin F. Juarez',
      author_email='ejuarez@ucsd.edu',
      license='MIT',
      packages=['cuscatlan'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
          ],
      )
