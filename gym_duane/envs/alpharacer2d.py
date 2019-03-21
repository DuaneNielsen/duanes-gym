import pygame
import pymunk
from pymunk import Vec2d
from pymunk.pygame_util import DrawOptions
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import weakref
from .events import EventQueue
from math import radians


class Gate:
    def __init__(self, space, x, total_height, gap_height, gap_size=50):
        top_length = total_height - gap_height - gap_size
        self.top_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.top = pymunk.Poly.create_box(self.top_body, (5, top_length))
        self.top.body.position = x, (top_length // 2) + gap_height + gap_size
        self.top.parent = weakref.ref(self)
        self.top.elasticity = 1.0
        space.add(self.top, self.top_body)

        self.opening_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.opening = pymunk.Poly.create_box(self.opening_body, (5, gap_size))
        self.opening.body.position = x, gap_height + (gap_size//2)
        self.opening.parent = weakref.ref(self)
        self.opening.sensor = True
        self.opening.elasticity = 1.0
        space.add(self.opening, self.opening_body)

        self.bottom_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.bottom = pymunk.Poly.create_box(self.bottom_body, (5, gap_height))
        self.bottom.body.position = x, gap_height // 2
        self.bottom.parent = weakref.ref(self)
        self.bottom.elasticity = 1.0

        space.add(self.bottom, self.bottom_body)

        self.space = space
        self.contacted = False

    def gate_contacted(self):
        """Returns true the first time this gate opening makes contact with the drone"""
        if not self.contacted:
            query_info = self.space.shape_query(self.opening)
            for object in query_info:
                if hasattr(object.shape, 'parent') and isinstance(object.shape.parent(), Drone):
                    self.contacted = True
                    return True
        return False


class Ground:
    def __init__(self, space, width):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, (width, 10))
        self.shape.body.position = width//2, 0
        self.shape.parent = weakref.ref(self)
        self.shape.elasticity = 1.0
        space.add(self.shape, self.body)


class Sky:
    def __init__(self, space, width, height):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, (width, 10))
        self.shape.body.position = width//2, height
        self.shape.parent = weakref.ref(self)
        self.shape.elasticity = 1.0
        space.add(self.shape, self.body)


class FinishLine:
    def __init__(self, space, width, height):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, (20, height))
        self.shape.body.position = width - 20, height // 2
        self.shape.parent = weakref.ref(self)
        self.shape.sensor = True
        self.space = space
        space.add(self.shape, self.body)

    def drone_enters(self):
        query_info = self.space.shape_query(self.shape)
        for object in query_info:
            if hasattr(object.shape, 'parent') and isinstance(object.shape.parent(), Drone):
                    return True
        else:
            return False


class DeathWall:
    def __init__(self, space, width, height, velocity):
        self.body = pymunk.Body(0, 0, pymunk.Body.KINEMATIC)
        self.shape = pymunk.Poly.create_box(self.body, (20, height - 20))
        self.shape.body.position = 0, height // 2
        self.shape.body.velocity = velocity
        self.shape.parent = weakref.ref(self)
        self.shape.sensor = False
        self.space = space
        space.add(self.shape, self.body)

    def drone_enters(self):
        query_info = self.space.shape_query(self.shape)
        for object in query_info:
            if hasattr(object.shape, 'parent') and isinstance(object.shape.parent(), Drone):
                    return True
        else:
            return False


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
        self.shape.parent = weakref.ref(self)
        self.space = env.space
        self.sensor = DepthSensor(self, env)
        self.space.add(self.body, self.shape)

    # Keep ball velocity at a static value
    def constant_velocity(self, body, gravity, damping, dt):
        self.shape.body.velocity = body.velocity.normalized() * 200

    def scan(self):
        start_angle = radians(45)
        arc = radians(90)/4
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

    def pulse(self, angle, depth, ticks=5):
        """
        Single raycast from drone
        :param angle: in radians
        :param depth: max distance sensor can detect
        :param ticks: number of ticks before deleting the ray
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

        self.env.event_q.add(self.clear_pulse, ticks)

        return distance

    def scan_arc(self, start_angle, arc, rays, depth):
        """
        Send a number of rays in an arc pattern from the body center of the drone
        start_angle is the first arc, subsequent rays are sent out at increments
        of arc in the clockwise direction
        :param start_angle: first angle to start from (radians)
        :param arc: increment for subsequent angles
        :param rays: number of rays
        :return: an array of distances to objects in the simulation
        """
        sensor_output = []
        for i in range(rays):
            angle = start_angle - i * arc
            sensor_output.append(self.pulse(angle=angle, depth=depth))
        sensor_output = np.array(sensor_output) / depth
        return sensor_output


class AlphaRacer2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.window = None
        self.width = 1200
        self.height = 400

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
        self.drone = Drone((50, self.height / 2), (-500, 0), self)

        self.gates = []
        self.gates.append(Gate(x=300, space=self.space, total_height=self.height, gap_height=50))
        self.gates.append(Gate(x=700, space=self.space, total_height=self.height, gap_height=300))
        self.gates.append(Gate(x=1100, space=self.space, total_height=self.height, gap_height=50))

        self.finishline = FinishLine(self.space, self.width, self.height)
        self.deathwall = DeathWall(self.space, self.width, self.height, (10, 0))

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
        ground = None
        for shape in arbiter.shapes:
            if hasattr(shape, 'parent'):
                if isinstance(shape.parent(), Drone):
                    drone = shape.parent
                elif isinstance(shape.parent(), Gate):
                    gate = shape.parent
                elif isinstance(shape.parent(), Ground):
                    ground = shape.parent
        if drone and (gate or ground):
            self.reward += -1.0
        return True

    def spawn_drone(self, dt):
        if self.drone:
            self.space.remove(self.drone.body, self.drone.shape)
        self.drone = Drone((50, self.height / 2), (0, 0), self)
        if self.deathwall:
            self.space.remove(self.deathwall.body, self.deathwall.shape)
        self.deathwall = DeathWall(self.space, self.width, self.height, (10, 0))

    def update(self, dt):
        self.space.step(dt)
        if self.finishline.drone_enters():
            self.done = True
            self.reward += 100
            self.spawn_drone(dt)
        if self.deathwall.drone_enters():
            self.done = True
            self.reward += -100
            self.spawn_drone(dt)
        for gate in self.gates:
            if gate.gate_contacted():
                self.reward += 100
        self.clock.tick()
        self.event_q.tick()

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

        self.reward = 0
        self.done = False

        self.event_q.execute()
        if self.display:
            pygame.event.pump()

        self.drone.action(action, modifiers=None)

        obs = self.step_simulation()

        return obs, self.reward, self.done, {}

    def step_simulation(self):

        # step the simulation
        for t in range(self.sim_steps):
            dt = self.step_time / self.sim_steps
            self.update(dt)

        dt = self.step_time / self.sim_steps
        self.update(dt)

        sensors = self.drone.scan()
        x = self.drone.shape.body.position.x / self.width
        y = self.drone.shape.body.position.y / self.height
        obs = np.concatenate((sensors, np.array([x, y])))

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
            if hasattr(shape, 'parent') and isinstance(shape.parent, Drone):
                self.spawn_drone(dt)
                self.done = False

        obs = self.step_simulation()

        return obs

    def render(self, mode='human'):
        if not self.display:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AlphaRacer")
            self.draw_options = DrawOptions(self.display)

        self.display.fill((0, 0, 0))
        self.space.debug_draw(self.draw_options)

        if mode is 'human':
            pygame.display.update()

        if mode is 'rgb_array':
            obs = pygame.surfarray.array3d(self.display)
            obs = obs.swapaxes(0, 1)
            return obs
