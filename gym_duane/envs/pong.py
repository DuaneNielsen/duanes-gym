import pygame
import pymunk
from pymunk.pygame_util import DrawOptions
from math import radians
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import cv2
import random


class PongEnv(gym.Env):
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
        self.vscreen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.draw_options = DrawOptions(self.vscreen)

        self.space = pymunk.Space()
        self.sim_steps = 10  # number of simulation steps per env step
        self.step_time = 0.05  # amount of simulation time per env step (seconds)

        self.player1 = Paddle(id='player1', position=(20, self.height / 2), angle=0, space=self.space,
                              window_height=self.height)
        self.player2 = Paddle(id='player2', position=(self.width - 20, self.height / 2),
                              angle=radians(180), space=self.space, window_height=self.height)
        self.puck = Puck((self.width / 2, self.height / 2), (-500, 0), self.space)

        self.top_rail = Rail(position=(self.width / 2, self.height - 20), space=self.space,
                             width=self.width)
        self.bottom_rail = Rail(position=(self.width / 2, 20), space=self.space, width=self.width)

        self.done = False
        self.reward = 0
        self.action_set = [0, 1, 2]
        self.action_space = spaces.Discrete(3)

        self.collision_handler = self.space.add_default_collision_handler()
        self.collision_handler.begin = self.coll_begin
        self.last_hit = None

    def coll_begin(self, arbiter, space, data):
        paddle = None
        puck = None
        for shape in arbiter.shapes:
            if isinstance(shape.thing, Paddle):
                paddle = shape.thing
            elif isinstance(shape.thing, Puck):
                puck = shape.thing
        if paddle and puck:
            self.last_hit = paddle.id
        return True

    def spawn_puck(self, dt):
        direction = 1.0 if random.random() < 0.5 else -1.0
        self.puck = Puck((self.width / 2, self.height / 2), (500 * direction, 0), self.space)
        self.last_hit = None

    def update(self, dt):
        for shape in self.space.shapes:
            if isinstance(shape.thing, Puck):
                if shape.body.position.x < 0 or shape.body.position.x > self.width:
                    self.space.remove(shape.body, shape)
                    self.done = True
                    if self.last_hit is 'player1':
                        self.reward = (1.0, -1.0)
                    elif self.last_hit is 'player2':
                        self.reward = (-1.0, 1.0)
                    else:
                        self.reward = (-1.0, -1.0)
                    self.spawn_puck(dt)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : (int, int) first int is player 1's action, second int is player 2's action
        0 -> UP, 1 -> Down, 2 - Stop
        

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

        self.reward = (0, 0)
        self.done = False
        player1_action, player2_action = self.action_set[action[0]], self.action_set[action[1]]

        self.player1.action(player1_action, modifiers=None)
        self.player2.action(player2_action, modifiers=None)

        obs, obs2 = self.step_simulation()

        return (obs, obs2), self.reward, self.done, {}

    def step_simulation(self):
        # step the simulation
        for t in range(self.sim_steps):
            dt = self.step_time / self.sim_steps
            self.space.step(dt)
            self.player1.update(dt)
            self.player2.update(dt)
            self.update(dt)
            self.clock.tick()

        self.vscreen.fill((0, 0, 0))
        self.space.debug_draw(self.draw_options)
        obs = pygame.surfarray.array3d(self.vscreen)
        obs = obs.swapaxes(0, 1)

        # now we have to oompute player 2's perspective
        transform = np.float32([[-1, 0, self.width],
                                [0, 1, 0]
                                ])
        obs2 = cv2.warpAffine(obs, transform, (self.width, self.height))
        return obs, obs2

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
                pygame.display.set_caption("Pong")
        self.display.blit(self.vscreen, (0, 0))
        pygame.display.update()


class Paddle:
    def __init__(self, id, position, angle, space, window_height):
        self.height = 30 * 3
        self.window_height = window_height
        self.w_1 = 2
        self.w_2 = 10
        self.id = id
        self.size = (10, 60)
        self.mass = 200000000
        self.body = pymunk.Body(self.mass, pymunk.inf, body_type=pymunk.Body.DYNAMIC)
        self.shape = pymunk.Poly(self.body, ((0, 0), (self.w_1, 0), (self.w_1 + self.w_2, self.height / 3),
                                             (self.w_1 + self.w_2, (self.height * 2) / 3), (self.w_1, self.height),
                                             (0, self.height)),
                                 transform=pymunk.Transform(tx=0, ty=-self.height / 2))
        self.shape.body.position = position
        self.shape.body.angle = angle
        self.shape.elasticity = 1.0
        self.shape.friction = 100.0
        self.shape.thing = self
        space.add(self.body, self.shape)

    def action(self, symbol, modifiers):
        if symbol == 0:
            self.body.velocity = 0, 500
        elif symbol == 1:
            self.body.velocity = 0, -500
        elif symbol == 2:
            self.body.velocity = 0, 0

    def get_action_meanings(self):
        return ['up', 'down', 'stop']

    def update(self, dt):
        pass
        # if self.shape.body.position.y < self.height / 2 + 20:
        #     self.shape.body.velocity = (0, 50)
        # if self.shape.body.position.y > self.window_height - self.height / 2 - 20:
        #     self.shape.body.velocity = (0, -50)


class Rail:
    def __init__(self, position, space, width):
        self.shape = pymunk.Poly.create_box(space.static_body, (width, 5))
        self.shape.body.position = position
        self.shape.thing = self
        self.shape.elasticity = 1.0
        space.add(self.shape)


class Puck:
    def __init__(self, position, velocity, space):
        self.mass = 1
        self.radius = 10
        self.moment = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.body.position = position
        self.shape.body.velocity = velocity
        self.shape.body.velocity_func = self.constant_velocity
        self.shape.elasticity = 1.0
        self.shape.friction = 100.0
        self.shape.thing = self
        self.space = space
        space.add(self.body, self.shape)

    # Keep ball velocity at a static value
    def constant_velocity(self, body, gravity, damping, dt):
        self.shape.body.velocity = body.velocity.normalized() * 500
