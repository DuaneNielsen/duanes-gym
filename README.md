# Duane's AI Gym

[![Build Status](https://travis-ci.org/DuaneNielsen/duanes-gym.svg?branch=master)](https://travis-ci.org/DuaneNielsen/duanes-gym)


This repository contains a PIP package which is an OpenAI environment for
a few simulations.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_duane

env = gym.make('AlphaRacer-v0')
```


## The Environments

- an adversarial Pong simulator

AlphaRacer2D-v0 - 2D drone simulation
PymunkPong-v0 - Adversarial Pong simulator