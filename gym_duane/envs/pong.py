import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
import pyglet.window.key as key
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
        self.window = pyglet.window.Window(800, 600, "Pymunk tester", resizable=False)
        self.options = DrawOptions()
        self.space = pymunk.Space()
        self.sim_steps = 3  # number of simulation steps per env step
        self.step_time = 0.03  # amount of simulation time per env step (seconds)

        @self.window.event
        def on_draw():
            self.window.clear()
            self.space.debug_draw(self.options)

        @self.window.event
        def on_key_press(symbol, modifiers):
            self.player1.on_key_press(symbol, modifiers)

        self.player1 = Paddle(id='player1', position=(20, self.window.height / 2), angle=0, space=self.space,
                              window_height=self.window.height)
        self.player2 = Paddle(id='player2', position=(self.window.width - 20, self.window.height / 2),
                              angle=radians(180), space=self.space, window_height=self.window.height)
        self.puck = Puck((self.window.width / 2, self.window.height / 2), (-500, 0), self.space)

        self.top_rail = Rail(position=(self.window.width / 2, self.window.height - 20), space=self.space,
                             width=self.window.width)
        self.bottom_rail = Rail(position=(self.window.width / 2, 20), space=self.space, width=self.window.width)

        self.done = False
        self.reward = 0
        self.action_set = [key.UP, key.DOWN, key.RIGHT]
        self.action_space = spaces.Discrete(3)

    def spawn_puck(self, dt):
        direction = 1.0 if random.random() < 0.5 else -1.0
        self.puck = Puck((self.window.width / 2, self.window.height / 2), (500 * direction, 0), self.space)

    def update(self, dt):
        for shape in self.space.shapes:
            if isinstance(shape.thing, Puck):
                if shape.body.position.x < 0 or shape.body.position.x > self.window.width:
                    self.space.remove(shape.body, shape)
                    self.spawn_puck(dt)
                    self.done = True
                if shape.body.position.x < 0:
                    # player 1 wins
                    self.reward = (1.0, -1.0)
                if shape.body.position.x > self.window.width:
                    # player 2 wins
                    self.reward = (-1.0, 1.0)

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

        self.player1.on_key_press(player1_action, modifiers=None)
        self.player2.on_key_press(player2_action, modifiers=None)

        # step the simulation
        for t in range(self.sim_steps):
            dt = self.step_time / self.sim_steps
            self.space.step(dt)
            self.player1.update(dt)
            self.player2.update(dt)
            self.update(dt)

        # grab the image from the framebuffer
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        # just take RGBA, there will be a SERIOUS performance penalty for converting format in pyglet
        # if you want to convert, use cv2.cvtColor() to do it
        image = image_data.get_data('RGBA', image_data.width * 4)
        obs = np.frombuffer(image, dtype=np.uint8)
        obs = obs.reshape(image_data.height, image_data.width, 4)

        # now we have to oompute player 2's perspective
        transform = np.float32([[-1, 0, image_data.width],
                                [0, 1, 0]
                                ])
        obs2 = cv2.warpAffine(obs, transform, (image_data.width, image_data.height))

        return (obs, obs2), self.reward, self.done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        ob = None

        return ob

    def render(self, mode='human'):
        if mode is 'human':
            pyglet.clock.tick()

            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event('on_draw')
                window.flip()

    @property
    def _n_actions(self):
        return len(self.action_set)


class Paddle:
    def __init__(self, id, position, angle, space, window_height):
        self.height = 30 * 3
        self.window_height = window_height
        self.w_1 = 2
        self.w_2 = 10
        self.id = id
        self.size = (10, 60)
        self.mass = 20000
        #self.moment = pymunk.moment_for_box(self.mass, self.size)
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

    def on_key_press(self, symbol, modifiers):
        if symbol == key.UP:
            self.body.velocity = 0, 500
        elif symbol == key.DOWN:
            self.body.velocity = 0, -500
        elif symbol == key.RIGHT:
            self.body.velocity = 0, 0

    def update(self, dt):
        if self.shape.body.position.y < self.height / 2 + 20:
            self.shape.body.velocity = (0, 50)
        if self.shape.body.position.y > self.window_height - self.height / 2 - 20:
            self.shape.body.velocity = (0, -50)


class Rail:
    def __init__(self, position, space, width):
        self.shape = pymunk.Poly.create_box(space.static_body, (width, 5))
        self.shape.body.position = position
        self.shape.thing = self
        self.shape.elasticity = 1.0
        space.add(self.shape)


class Puck:
    def __init__(self, position, velocity, space):
        self.mass = 5
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
