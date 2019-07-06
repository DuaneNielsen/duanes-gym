import gym
import torch
from colorama import Style, Fore, Back
from lark import Lark
from torch.nn.functional import one_hot

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
NOP = 4

lander = """
[
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[E],
[E],
[E],
[E],
[S],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[E],
[T(1.0)],
[T(0.5)],
[T(0.25)],
[T(0.25)],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L],
[L]
]
"""


class LunarLander(gym.Env):
    def __init__(self, n, device):
        super().__init__()
        l = Lark(
            '''
            start: "[" row ("," row )* "]"
            row: "[" token ("," token)* "]" 
            token: WORD ( "(" SIGNED_NUMBER ")" )?

            %import common.WORD
            %import common.SIGNED_NUMBER
            %import common.NEWLINE
            %ignore " " | NEWLINE 
            ''')

        tree = l.parse(lander)

        height = len(tree.children)
        width = len(tree.children[0].children)

        self.observation_space_shape = ((height, width), (20,))
        self.action_space = gym.spaces.Discrete(4)
        self.device = device

        self.gravity = 1
        self.thrust = -3

        with torch.no_grad():
            self.map = torch.zeros(n, height, width, dtype=torch.long, requires_grad=False, device=device)
            self.position_y = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.position_x = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.terminated = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.terminal = torch.zeros(height, width, dtype=torch.long, requires_grad=False, device=device)
            self.rewards = torch.zeros(height, width, dtype=torch.float32, requires_grad=False, device=device)
            self.range = torch.arange(n, device=device)
            self.height = height
            self.width = width
            self.start = (0, 0)
            self.n = n

            self.speed = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)

            def elem(*args):
                terminal_state = 0
                reward = 0.0 if len(args) == 1 else float(args[1].value)
                if args[0] == 'L':
                    reward = -1.0
                if args[0] == 'T' or args[0] == 'L':
                    terminal_state = 1.0
                is_start = args[0] == 'S'
                return reward, terminal_state, is_start

            for i, row in enumerate(tree.children):
                for j, token in enumerate(row.children):
                    reward, terminal_state, is_start = elem(*token.children)
                    self.terminal[i, j] = terminal_state
                    self.rewards[i, j] = reward
                    if is_start:
                        self.start_x = torch.tensor([j], device=self.device)
                        self.start_y = torch.tensor([i], device=self.device)

            self.position_y = self.start_x.repeat(self.n)
            self.position_x = self.start_y.repeat(self.n)

            self.t = torch.tensor([[-1.0, 0.0],
                                   [1.0, 0.0],
                                   [0.0, 0.0],
                                   [0.0, 0.0],
                                   [0.0, 0.0],
                                   ], requires_grad=False, device=device)

    def position_index(self):
        # compute and write the position
        with torch.no_grad():
            base = torch.arange(self.n, device=self.device) * self.height * self.width
            offset = self.position_y * self.width + self.position_x
            index = base + offset
            return index

    def update_map(self):
        self.map.zero_()
        self.map.flatten()[self.position_index()] = 1.0

    def reset(self):
        with torch.no_grad():
            self.speed.zero_()
            self.position_x = self.start_x.repeat(self.n)
            self.position_y = self.start_y.repeat(self.n)
            self.terminated.zero_()
            self.update_map()
            return self.observation

    def reset_done(self):
        with torch.no_grad():
            index = torch.masked_select(torch.arange(self.terminated.size(0), device=self.device), self.terminated.to(dtype=torch.uint8))
            self.speed[index].zero_()
            self.position_x[index] = self.start_x
            self.position_y[index] = self.start_y
            self.terminated.zero_()

    @property
    def observation(self):
        map = self.map.to(dtype=torch.float32, device=self.device)
        speed = one_hot(self.speed + 10, 20).to(dtype=torch.float32, device=self.device)
        return map, speed

    def step(self, actions):
        with torch.no_grad():
            self.speed += self.gravity
            self.speed[actions.eq(UP)] += self.thrust
            self.speed.clamp(-10, 10)
            actions = one_hot(actions, 5).float()
            actions = actions.matmul(self.t).long()
            self.position_x += actions[:, 0]
            self.position_y += self.speed
            self.position_x.clamp_(0, self.width - 1)
            self.position_y.clamp_(0, self.height - 1)
            self.reset_done()
            self.update_map()
            reward = self.rewards.unsqueeze(0).expand(self.n, -1, -1).flatten()[self.position_index()]
            self.terminated = torch.sum(self.map & self.terminal, dim=(1, 2))
            done = self.terminated.to(dtype=torch.uint8, device=self.device)
            return self.observation, reward, done, {}

    def render(self, mode='human'):

        offset = self.position_y * self.width + self.position_x
        values, counts = torch.unique(offset, return_counts=True)
        bins = torch.zeros(self.height * self.width, device=self.device)
        bins[values] = counts.float()
        bins = bins / torch.sum(bins)
        bins = bins.reshape(self.height, self.width)
        mean = bins[bins != 0.0].float().mean().item()
        small = 0.01

        s = '\n'
        for i, row in enumerate(bins):
            s = ''
            for j, column in enumerate(row):
                v = bins[i, j].item()
                c = "{0:02.0f}".format(v*100)
                color = Fore.BLUE
                if v == 0:
                    color = Fore.BLACK
                if mean < v:
                    color = Fore.RED
                if mean >= v >= small:
                    color = Fore.MAGENTA

                c = f'{Back.BLACK}{color}{Style.BRIGHT}{c}{Style.RESET_ALL}'
                s = s + c
            print(s)