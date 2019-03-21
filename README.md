# Duane's AI Gym

[![Build Status](https://travis-ci.org/DuaneNielsen/duanes-gym.svg?branch=master)](https://travis-ci.org/DuaneNielsen/duanes-gym)


This repository contains a PIP package which is an OpenAI environment for
a few simulations.


## Installation

Install this package from Pypi

```
pip install gym-duane
```

Or if from source, then

```
git clone https://github.com/DuaneNielsen/duanes-gym.git
cd duanes-gym
pip install -e .

```

## The Environments

AlphaRacer2D-v0 - 2D drone simulation

```
import gym
import gym_duane

env = gym.make('AlphaRacer2D-v0')
```


PymunkPong-v0 - Adversarial Pong simulator

```
import gym
import gym_duane

env = gym.make('PymunkPong-v0')
```
