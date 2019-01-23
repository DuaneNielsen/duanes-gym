#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_duane
import time

class Environments(unittest.TestCase):

    def test_bannana(self):
        env = gym.make('Banana-v0')
        env.seed(0)
        env.reset()
        env.step(0)

    def test_pong(self):
        env = gym.make('PymunkPong-v0')
        env.seed(0)
        env.reset()
        env.step(0)

    def test_pong_loop(self):
        env = gym.make('PymunkPong-v0')

        for _ in range(3000):
            observation = env.step(0)
            env.render()