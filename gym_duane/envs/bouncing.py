import pygame
import pymunk
from pymunk import Vec2d
from pymunk.pygame_util import DrawOptions
from math import radians
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import cv2
import random
from math import degrees


class Paddle:
    def __init__(self, id, position, angle, space, window_height):
        self.height = 30 * 3
        self.window_height = window_height
        self.w_1 = 10
        self.w_2 = 10
        self.id = id
        self.size = (10, 60)
        self.mass = 20
        self.body = pymunk.Body(self.mass, pymunk.inf, body_type=pymunk.Body.KINEMATIC)
        self.shape = pymunk.Poly(self.body, ((0, 0), (self.w_1, 0),
                                             (self.w_1, self.height),
                                             (0, self.height)),
                                 transform=pymunk.Transform(tx=0, ty=-self.height / 2))
        self.shape.body.position = position
        self.shape.body.angle = angle
        self.shape.elasticity = 1.0
        self.shape.friction = 100.0
        self.shape.thing = self
        space.add(self.body, self.shape)


        self.velocity = [Vec2d(0, 0), Vec2d(1, 0), Vec2d(-1, 0), Vec2d(0, -1), Vec2d(0, 1)]

    # def action(self, power, angle, modifiers):
    #     impulse = power * Vec2d(1, 0)
    #     impulse.rotate(angle)
    #     self.body.apply_impulse_at_local_point(impulse)

    def action(self, action, modifiers):
        self.body.velocity = self.velocity[action] * 300

    def get_action_meanings(self):
        return ['stop', 'up', 'down', 'left', 'right']


class Puck:
    def __init__(self, position, velocity, space):
        self.mass = 1
        self.radius = 10
        self.moment = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.body.position = position
        self.shape.body.velocity = velocity
        self.shape.elasticity = 1.0
        self.shape.friction = 100.0
        self.shape.thing = self
        self.space = space
        space.add(self.body, self.shape)


class BounceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.window = None
        self.width = 800
        self.height = 600

        # init framebuffer for observations
        pygame.init()
        self.display = None
        self.draw_options = None
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = 0, -1000
        self.sim_steps = 10  # number of simulation steps per env step
        self.step_time = 0.05  # amount of simulation time per env step (seconds)


        self.puck = None
        self.spawn_puck(0)

        self.paddle = None
        self.spawn_paddle(0)

        self.done = False
        self.reward = 0

    def spawn_puck(self, dt):
        self.puck = Puck((self.width / 2, self.height - 20), velocity=(0, 0), space=self.space)

    def spawn_paddle(self, dt):
        if self.paddle is not None:
            self.space.remove(self.paddle.body, self.paddle.shape)
        self.paddle = Paddle(id='player1', position=(500, 40), angle=radians(90), space=self.space,
                             window_height=self.height)

    def update(self, dt):
        for shape in self.space.shapes:
            if isinstance(shape.thing, Puck):
                if shape.body.position.x < 0 or shape.body.position.x > self.width or shape.body.position.y < 0:
                    self.space.remove(shape.body, shape)
                    self.done = True
                    self.spawn_puck(dt)
                    self.spawn_paddle(dt)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : force to apply to puck - 1D numpy array of shape (2)  (power, rotation)

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) : pixels from the screen, dont forget to put the mirror image of
            the screen to player 2, or he will get confused!
            reward (float, float) : a tuple of floats, (player1 reward, player2 reward)
            episode_over (bool) :
                the ball went out of play,
            info (dict) :
                 empty at the moment
        """

        self.reward = 1
        self.done = False
        # power = action.item(0)
        # angle = action.item(1)

        #self.player1.action(power, angle, modifiers=None)
        self.paddle.action(action, modifiers=None)

        obs = self.step_simulation()

        return obs, self.reward, self.done, {}

    def step_simulation(self):
        # step the simulation
        for t in range(self.sim_steps):
            dt = self.step_time / self.sim_steps
            self.space.step(dt)
            # self.player1.update(dt)
            self.update(dt)
            self.clock.tick()

        if self.display:
            self.display.fill((0, 0, 0))
            self.space.debug_draw(self.draw_options)

        obs = np.array([self.paddle.body.position.x,
                        self.paddle.body.position.y,
                        self.paddle.body.rotation_vector.x,
                        self.paddle.body.rotation_vector.y,
                        self.puck.body.position.x,
                        self.puck.body.position.y])

        return obs

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        dt = self.step_time / self.sim_steps
        for shape in self.space.shapes:
            if isinstance(shape.thing, Puck):
                self.space.remove(shape.body, shape)
                self.spawn_puck(dt)
                self.done = False

        ob = self.step_simulation()

        return ob

    def render(self, mode='human'):
        if mode is 'human':
            if not self.display:
                self.display = pygame.display.set_mode((self.width, self.height))
                self.draw_options = DrawOptions(self.display)
                pygame.display.set_caption("Pong")
        pygame.display.flip()
