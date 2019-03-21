#!/usr/bin/env python
from setuptools import setup

setup(name='gym_duane',
      version='0.0.1',
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
