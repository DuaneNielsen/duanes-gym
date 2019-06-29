import torch
import gym
import logging
from lark import Lark
from torch.nn.functional import one_hot

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self):
        self.parser = Lark(
            '''
            start: "[" row "]"
            row: token ("," token)*
            token: WORD ( "(" SIGNED_NUMBER ")" )?

            %import common.WORD
            %import common.SIGNED_NUMBER
            %ignore " "           // Disregard spaces in text
            ''')

        self.start = 0
        self.row = 0
        self.shape = 0
        self.rewards = []
        self.terminal = []

    def element(self, *args):
        self.shape += 1
        reward = 0.0 if len(args) == 1 else float(args[1].value)
        terminal_state = 1 if args[0] == 'T' else 0
        if args[0] == 'S':
            self.start = self.row
        self.rewards.append(reward)
        self.terminal.append(terminal_state)
        self.row += 1

    def parse(self, map_string):
        tree = self.parser.parse(map_string)
        for token in tree.children[0].children:
            print(*token.children)
            self.element(*token.children)


class SimpleGrid(gym.Env):
    def __init__(self, n, map_string):
        super().__init__()
        self.parser = Parser()
        self.parser.parse(map_string)
        shape = self.parser.shape
        self.map = torch.zeros(n, shape, dtype=torch.long, requires_grad=False)
        self.position = torch.tensor([self.parser.start] * n, dtype=torch.long, requires_grad=False)
        self.terminated = torch.zeros(n, dtype=torch.long, requires_grad=False)
        self.terminal = torch.tensor(self.parser.terminal, dtype=torch.long, requires_grad=False)
        self.rewards = torch.tensor(self.parser.rewards, dtype=torch.float16, requires_grad=False)
        self.range = torch.arange(n)
        self.shape = shape
        self.row = 0

    def reset(self):
        with torch.no_grad():
            self.position.fill_(self.parser.start)
            self.map[self.range, self.position] = 1.0
            self.terminated.zero_()

    def step(self, actions):
        with torch.no_grad():
            torch
            actions = actions.to(dtype=torch.long)
            actions = actions * (1 - self.terminated)
            self.position += actions
            self.position.clamp_(0, self.shape - 1)
            self.map.zero_()
            self.map[self.range, self.position] = 1
            reward = self.rewards[self.position]
            self.terminated = torch.sum(self.map & self.terminal, dim=(1,))
            return self.map, reward, self.terminated

    def render(self, mode='human'):
        print(f'{self.map.data}')


class SimpleGridV2(gym.Env):
    def __init__(self, n, map_string):
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

        tree = l.parse(map_string)

        height = len(tree.children)
        width = len(tree.children[0].children)

        if height > 1 and width > 1:
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Discrete(2)

        with torch.no_grad():
            self.map = torch.zeros(n, height, width, dtype=torch.long, requires_grad=False)
            self.position_y = torch.zeros(n, dtype=torch.long, requires_grad=False)
            self.position_x = torch.zeros(n, dtype=torch.long, requires_grad=False)
            self.terminated = torch.zeros(n, dtype=torch.long, requires_grad=False)
            self.terminal = torch.zeros(height, width, dtype=torch.long, requires_grad=False)
            self.rewards = torch.zeros(height, width, dtype=torch.float32, requires_grad=False)
            self.range = torch.arange(n)
            self.height = height
            self.width = width
            self.start = (0, 0)
            self.n = n

            def elem(*args):
                reward = 0.0 if len(args) == 1 else float(args[1].value)
                terminal_state = 1 if args[0] == 'T' else 0
                is_start = args[0] == 'S'
                return reward, terminal_state, is_start

            for i, row in enumerate(tree.children):
                for j, token in enumerate(row.children):
                    reward, terminal_state, is_start = elem(*token.children)
                    self.terminal[i, j] = terminal_state
                    self.rewards[i, j] = reward
                    if is_start:
                        self.start_x = torch.tensor([i])
                        self.start_y = torch.tensor([j])

            self.position_y = self.start_x.repeat(self.n)
            self.position_x = self.start_y.repeat(self.n)
            self.t = torch.tensor([[-1, 0],
                                   [1, 0],
                                   [0, -1],
                                   [0, 1]
                                   ], requires_grad=False)

    def position_index(self):
        # compute and write the position
        with torch.no_grad():
            base = torch.arange(self.n) * self.height * self.width
            offset = self.position_y * self.width + self.position_x
            index = base + offset
            return index

    def update_map(self):
        self.map.zero_()
        self.map.flatten()[self.position_index()] = 1.0

    def reset(self):
        with torch.no_grad():
            self.position_y = self.start_x.repeat(self.n)
            self.position_x = self.start_y.repeat(self.n)
            self.terminated.zero_()
            self.update_map()
            return self.map.to(dtype=torch.float32)

    def step(self, actions):
        with torch.no_grad():
            actions = one_hot(actions, 4)
            actions = actions.matmul(self.t)
            actions = actions * (1 - self.terminated).unsqueeze(1).expand(-1, 2)
            self.position_x += actions[:, 0]
            self.position_y += actions[:, 1]
            self.position_x.clamp_(0, self.width - 1)
            self.position_y.clamp_(0, self.height - 1)
            self.update_map()
            reward = self.rewards.unsqueeze(0).expand(10, -1, -1).flatten()[self.position_index()]
            self.terminated = torch.sum(self.map & self.terminal, dim=(1, 2))
            return self.map.to(dtype=torch.float32), reward, self.terminated.to(dtype=torch.uint8)

    def render(self, mode='human'):
        print(f'{self.map.data}')
