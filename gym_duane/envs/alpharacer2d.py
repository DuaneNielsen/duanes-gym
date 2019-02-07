import pygame
import pymunk
from pymunk import Vec2d, Transform
from pymunk.pygame_util import DrawOptions, to_pygame
from math import radians
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import cv2
import random
import math
from threading import Timer
import weakref
from common.events import EventQueue, Event
from math import degrees


class Gate:
    def __init__(self, space, x, total_height, gap_height, gap_size=50):
        top_length = total_height - gap_height - gap_size
        self.top = pymunk.Poly.create_box(space.static_body, (5, top_length))
        self.top.body.position = x, (top_length // 2) + gap_height + gap_size
        self.top.thing = weakref.ref(self)
        self.top.elasticity = 1.0
        space.add(self.top)

        self.bottom = pymunk.Poly.create_box(space.static_body, (5, gap_height))
        self.bottom.body.position = x, gap_height // 2
        self.bottom.thing = weakref.ref(self)
        self.bottom.elasticity = 1.0

        space.add(self.bottom)


class Ground:
    def __init__(self, space, width):
        self.shape = pymunk.Poly.create_box(space.static_body, (width, 10))
        self.shape.body.position = width//2, 0
        self.shape.thing = self
        self.shape.elasticity = 1.0
        space.add(self.shape)


class Sky:
    def __init__(self, space, width, height):
        self.shape = pymunk.Poly.create_box(space.static_body, (width, 10))
        self.shape.body.position = width//2, height
        self.shape.thing = weakref.ref(self)
        self.shape.elasticity = 1.0
        space.add(self.shape)


class Drone:
    def __init__(self, position, velocity, env):
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
        self.shape.filter = pymunk.ShapeFilter(categories=0b1)
        self.shape.thing = self
        self.space = env.space
        self.sensor = DepthSensor(self, env)
        self.space.add(self.body, self.shape)

    # Keep ball velocity at a static value
    def constant_velocity(self, body, gravity, damping, dt):
        self.shape.body.velocity = body.velocity.normalized() * 200

    def scan(self):
        start_angle = degrees(45)
        arc = degrees(45/5)
        depth = 400
        return self.sensor.scan_arc(start_angle, arc, 5, depth)

    def action(self, action, modifiers):
        if action == 0:
            self.body.velocity = 0, 200
        elif action == 1:
            self.body.velocity = 0, -200
        elif action == 2:
            self.body.velocity = -200, 0
        elif action == 3:
            self.body.velocity = 200, 0


class DepthSensor:
    def __init__(self, drone, env):
        self.drone = drone
        self.space = env.space
        self.env = env

    def clear_pulse(self):
        for shape in self.space.shapes:
            if shape.sensor:
                self.space.remove(shape)

    def pulse(self, angle, depth):
        """
        Single raycast from drone
        :param angle: in radians
        :param depth: max distance sensor can detect
        """
        origin = self.drone.body.position
        end = Vec2d(1, 0)
        end.rotate(angle + self.drone.body.angle)
        end.length = depth
        end = end + origin

        segment_query = self.space.segment_query_first(origin, end, 0, pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS ^ 0b1))
        if segment_query:
            contact_point = segment_query.point
            line = pymunk.Segment(self.space.static_body, origin, contact_point, 1)
            line.sensor = True
            line.body.position = 0, 0
            self.space.add(line)
            distance = (origin - contact_point).length
        else:
            line = pymunk.Segment(self.space.static_body, origin, end, 1)
            line.sensor = True
            line.body.position = 0, 0
            self.space.add(line)
            distance = depth

        self.env.event_q.add(self.clear_pulse, 10)

        return distance

    def scan_arc(self, start_angle, arc, rays, depth):
        """
        Send a number of rays in an arc pattern from the body center of the drone
        :param start_angle: first angle to start from (radians)
        :param arc: increment for subsequent angles
        :param rays: number of rays
        :return: an array of distances to objects in the simulation
        """
        sensor_output = []
        for i in range(rays):
            angle = i * arc + start_angle
            sensor_output.append(self.pulse(angle=angle, depth=depth))
        return sensor_output


class AlphaRacer2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.window = None
        self.width = 1200
        self.height = 600

        self.event_q = EventQueue()

        # init framebuffer for observations
        pygame.init()
        self.display = None
        self.clock = pygame.time.Clock()
        self.draw_options = None

        self.space = pymunk.Space()
        self.sim_steps = 10  # number of simulation steps per env step
        self.step_time = 0.05  # amount of simulation time per env step (seconds)

        self.ground = Ground(self.space, self.width)
        self.sky = Sky(self.space, self.width, self.height)
        self.drone = Drone((self.width / 2, self.height / 2), (-500, 0), self)

        self.first = Gate(x=200, space=self.space, total_height=self.height, gap_height=300)

        self.done = False
        self.reward = 0
        self.action_set = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)

        self.collision_handler = self.space.add_default_collision_handler()
        self.collision_handler.begin = self.coll_begin
        self.last_hit = None

    def coll_begin(self, arbiter, space, data):
        drone = None
        gate = None
        for shape in arbiter.shapes:
            if hasattr(shape, 'thing') and isinstance(shape.thing, Drone):
                drone = shape.thing
            elif hasattr(shape, 'thing') and isinstance(shape.thing, Gate):
                gate = shape.thing
        if drone and gate:
            self.reward = -1.0
        return True

    def spawn_drone(self, dt):
        direction = 1.0 if random.random() < 0.5 else -1.0
        self.drone = Drone((self.width / 2, self.height / 2), (0, 0), self)
        self.last_hit = None

    def update(self, dt):
        for shape in self.space.shapes:
            if hasattr(shape, 'thing') and isinstance(shape.thing, Drone):
                if shape.body.position.x < 0 or shape.body.position.x > self.width:
                    self.space.remove(shape.body, shape)
                    self.done = True
                    self.reward = 1.0 if shape.body.position.x < 0 else -1.0
                    self.spawn_drone(dt)

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

        self.event_q.execute()
        pygame.event.pump()

        self.drone.action(action, modifiers=None)

        obs = self.step_simulation()

        return obs, self.reward, self.done, {}

    def step_simulation(self):

        # step the simulation
        for t in range(self.sim_steps):
            dt = self.step_time / self.sim_steps
            self.space.step(dt)
            #self.drone.update(dt)
            self.update(dt)
            self.clock.tick()
            self.event_q.tick()

        dt = self.step_time / self.sim_steps
        self.space.step(dt)
        obs = self.drone.scan()
        self.update(dt)
        self.clock.tick()
        self.event_q.tick()
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
            if hasattr(shape, 'thing') and isinstance(shape.thing, Drone):
                self.space.remove(shape.body, shape)
                self.spawn_drone(dt)
                self.done = False

        ob = self.step_simulation()

        return ob

    def render(self, mode='human'):
        if not self.display:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AlphaRacer")
            self.draw_options = DrawOptions(self.display)

        self.display.fill((0, 0, 0))
        self.space.debug_draw(self.draw_options)

        if mode is 'human':
            pygame.display.flip()

        if mode is 'rgb_array':
            obs = pygame.surfarray.array3d(self.display)
            obs = obs.swapaxes(0, 1)
            return obs
