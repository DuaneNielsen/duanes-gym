#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='gym_duane',
      version='0.0.7',
      packages=find_packages(),
      install_requires=['gym>=0.2.3',
                        'pymunk',
                        'pygame',
                        'opencv-python',
                        'numpy',
                        'lark-parser'],
      extras_require={
          'dev': [
              'pytest',
              'tox',
              'unittest2'
          ]
      }
      )
