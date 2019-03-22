#!/usr/bin/env python
from setuptools import setup, find_namespace_packages

setup(name='gym_duane',
      version='0.0.4',
      packages=find_namespace_packages(include=['gym_duane.*']),
      install_requires=['gym>=0.2.3',
                        'pymunk',
                        'pygame',
                        'opencv-python',
                        'numpy'],
      extras_require={
          'dev': [
              'pytest',
              'tox',
              'unittest2'
          ]
      }
      )
