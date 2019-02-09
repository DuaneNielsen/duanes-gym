#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import unittest

# 3rd party modules
import gym
# from .viewer import UniImageViewer

# internal modules
import gym_duane
import time

import cv2
import numpy as np


def to_numpyRGB(image, invert_color=False):
    """
    Universal method to detect and convert an image to numpy RGB format
    :params image: the output image
    :params invert_color: perform RGB -> BGR convert
    :return: the output image
    """
    if type(image) == 'torch.Tensor':
        image = image.cpu().detach().numpy()
    # remove batch dimension
    if len(image.shape) == 4:
        image = image[0]
    smallest_index = None
    if len(image.shape) == 3:
        smallest = min(image.shape[0], image.shape[1], image.shape[2])
        smallest_index = image.shape.index(smallest)
    elif len(image.shape) == 2:
        smallest = 0
    else:
        raise Exception(f'too many dimensions, I got {len(image.shape)} dimensions, give me less dimensions')
    if smallest == 3:
        if smallest_index == 2:
            pass
        elif smallest_index == 0:
            image = np.transpose(image, [1, 2, 0])
        elif smallest_index == 1:
            # unlikely
            raise Exception(f'Is this a color image?')
        if invert_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif smallest == 1:
        image = np.squeeze(image)
    elif smallest == 0:
        # greyscale
        pass
    elif smallest == 4:
        # that funny format with 4 color dims
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        raise Exception(f'dont know how to display color of dimension {smallest}')
    return image


class UniImageViewer:
    def __init__(self, title='title', screen_resolution=(640, 480), format=None, channels=None, invert_color=True):
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution
        self.format = format
        self.channels = channels
        self.invert_color = invert_color

    def render(self, image, block=False):

        image = to_numpyRGB(image, self.invert_color)

        image = cv2.resize(image, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, image)
        if block:
            cv2.waitKey(0)
            pass
        else:
            cv2.waitKey(1)
            pass

    def view_input(self, model, input, output):
        image = input[0] if isinstance(input, tuple) else input
        self.render(image)

    def view_output(self, model, input, output):
        image = output[0] if isinstance(output, tuple) else output
        self.render(image)

    def update(self, image):
        self.render(image)


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
        try:
            env = gym.make('PymunkPong-v0')
            p1 = UniImageViewer('player1')
            p2 = UniImageViewer('player2')

            for game in range(50):
                print(f'starting game {game}')
                obs = env.reset()
                done = False
                p1_reward = 0
                p2_reward = 0

                while not done:
                    actions = env.action_space.sample(), env.action_space.sample()
                    observation, reward, done, info = env.step(actions)
                    p1.render(observation[0], block=False)
                    p2.render(observation[1], block=False)
                    # env.render()

                    p1_reward, p2_reward = reward

                if p1_reward > p2_reward:
                    print('player1 won')
                if p1_reward < p2_reward:
                    print('player2 won')
                else:
                    print('nobody won')
        except Exception as e:
            print(e)

    def test_racerloop(self):
        env = gym.make('AlphaRacer2D-v0')
        p1 = UniImageViewer('player1')

        for game in range(50):
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                actions = env.action_space.sample()
                observation, reward, done, info = env.step(actions)
                env.render()
                time.sleep(0.05)
                total_reward += reward
                print(reward)

            print(f'game {game} ended with reward {total_reward}')

    def test_EventQueue(self):
        from envs.events import Event, EventQueue
        q = EventQueue()

        def hello_callback(arg, kw_arg='yep'):
            print(f'hello {arg} {kw_arg}' )

        q.add(Event(hello_callback, 'from the ', 'event'))

        for event in q:
            event.execute()

        class J:
            def __init__(self):
                self.j = 0

            def plus_one(self):
                self.j += 1

            def plus_two(self):
                self.j += 2


        j = J()

        q.add(Event(j.plus_one))
        q.add(Event(j.plus_one))
        q.add(Event(j.plus_one))

        for event in q:
            event.execute()

        assert j.j == 3

        j.j = 0

        q.add(Event(j.plus_two), 5)
        q.add(Event(j.plus_one), 2)
        q.add(Event(j.plus_one))

        for event in q:
            event.execute()

        assert j.j == 1

        q.tick()

        for event in q:
            event.execute()

        assert j.j == 1

        q.tick()

        for event in q:
            event.execute()

        assert j.j == 2

        q.tick()

        for event in q:
            event.execute()

        assert j.j == 2

        q.tick()
        q.tick()

        for event in q:
            event.execute()

        assert j.j == 4

