import torch
import gym
import logging
from lark import Lark
from torch.nn.functional import one_hot
from colorama import Style, Fore, Back

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
            actions = actions.to(dtype=torch.long)
            actions = actions * (1 - self.terminated)
            self.position += actions
            self.position.clamp_(0, self.shape - 1)
            self.map.zero_()
            self.map[self.range, self.position] = 1
            reward = self.rewards[self.position]
            self.terminated = torch.sum(self.map & self.terminal, dim=(1,))
            return self.map, reward, self.terminated, {}

    def render(self, mode='human'):
        print(f'{self.map.data}')


class SimpleGridV2(gym.Env):
    def __init__(self, n, map_string, device):
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

        self.height = len(tree.children)
        self.width = len(tree.children[0].children)

        self.observation_space_shape = ((self.height, self.width),)
        self.action_space = gym.spaces.Discrete(4)
        self.device = device

        with torch.no_grad():
            self.map = torch.zeros(n, self.height, self.width, dtype=torch.long, requires_grad=False, device=device)
            self.position_y = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.position_x = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.terminated = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.terminal = torch.zeros(self.height, self.width, dtype=torch.long, requires_grad=False, device=device)
            self.rewards = torch.zeros(self.height, self.width, dtype=torch.float32, requires_grad=False, device=device)
            self.range = torch.arange(n, device=device)
            self.start = (0, 0)
            self.n = n

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
                                   [0.0, -1.0],
                                   [0.0, 1.0]
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
            self.position_x = self.start_x.repeat(self.n)
            self.position_y = self.start_y.repeat(self.n)
            self.terminated.zero_()
            self.update_map()
            return self.map.to(dtype=torch.float32)

    def reset_done(self):
        index = torch.masked_select(torch.arange(self.terminated.size(0), device=self.device),
                                    self.terminated.to(dtype=torch.uint8))
        self.position_x[index] = self.start_x
        self.position_y[index] = self.start_y
        self.terminated.zero_()

    def step(self, actions):
        with torch.no_grad():
            actions = one_hot(actions, 4).float()
            actions = actions.matmul(self.t).long()
            self.position_x += actions[:, 0]
            self.position_y += actions[:, 1]
            self.position_x.clamp_(0, self.width - 1)
            self.position_y.clamp_(0, self.height - 1)
            self.reset_done()
            self.update_map()
            reward = self.rewards.unsqueeze(0).expand(self.n, -1, -1).flatten()[self.position_index()]
            self.terminated = torch.sum(self.map & self.terminal, dim=(1, 2))
            return self.map.to(dtype=torch.float32, device=self.device), reward, self.terminated.to(dtype=torch.uint8,
                                                                                                    device=self.device), {}

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
                c = "{0:02.0f}".format(v * 100)
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


class SimpleGridV3(gym.Env):
    def __init__(self, n, map_string, device, max_steps, reward_per_timestep=-0.0):
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

        self.height = len(tree.children)
        self.width = len(tree.children[0].children)

        self.observation_space_shape = (self.height, self.width)
        self.action_space = gym.spaces.Discrete(4)
        self.device = device
        self.max_steps = max_steps

        with torch.no_grad():
            self.map = torch.zeros(n, self.height, self.width, dtype=torch.long, requires_grad=False, device=device)
            self.position_y = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.position_x = torch.zeros(n, dtype=torch.long, requires_grad=False, device=device)
            self.done = torch.zeros(n, dtype=torch.uint8, requires_grad=False, device=device)
            self._reset = torch.zeros(n, dtype=torch.uint8, requires_grad=False, device=device)
            self.terminal = torch.zeros(self.height, self.width, dtype=torch.long, requires_grad=False, device=device)
            self.rewards = torch.zeros(self.height, self.width, dtype=torch.float32, requires_grad=False, device=device)
            self.reward_present = None
            self.reward_present_init = torch.zeros(n, self.height, self.width, dtype=torch.uint8, requires_grad=False,
                                                   device=device)
            self.range = torch.arange(n, device=device)
            self.step_counter = torch.zeros(n, dtype=torch.int16, device=device)
            self.start = (0, 0)
            self.n = n
            self.reward_per_timestep = reward_per_timestep

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
                    self.reward_present_init[:, i, j] = True
                    if is_start:
                        self.start_x = torch.tensor([j], device=self.device)
                        self.start_y = torch.tensor([i], device=self.device)

            self.position_y = self.start_x.repeat(self.n)
            self.position_x = self.start_y.repeat(self.n)
            self.t = torch.tensor([[-1.0, 0.0],
                                   [1.0, 0.0],
                                   [0.0, -1.0],
                                   [0.0, 1.0]
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
            self.position_x = self.start_x.repeat(self.n)
            self.position_y = self.start_y.repeat(self.n)
            self.reward_present = self.reward_present_init.clone()
            self.done.zero_()
            self._reset.zero_()
            self.update_map()
            return self.map.to(dtype=torch.float32)

    def reset_done(self):
        done = torch.masked_select(torch.arange(self.done.size(0), device=self.device),
                                   self.done)

        self.position_x[done] = self.start_x
        self.position_y[done] = self.start_y
        self.reward_present[done] = self.reward_present_init[done]
        self.step_counter[done] = 0
        self._reset.zero_()
        self._reset[done] = 1
        self.done.zero_()

    def step(self, actions):
        with torch.no_grad():

            actions = one_hot(actions, 4).float()
            actions = actions.matmul(self.t).long()
            self.position_x += actions[:, 0]
            self.position_y += actions[:, 1]
            self.position_x.clamp_(0, self.width - 1)
            self.position_y.clamp_(0, self.height - 1)
            self.step_counter += 1

            self.reset_done()
            self.update_map()

            # give rewards if not already given
            reward = self.rewards.unsqueeze(0).expand(self.n, -1, -1).flatten()[self.position_index()]
            reward = reward * self.reward_present.flatten()[self.position_index()].float()
            reward += self.reward_per_timestep
            self.reward_present.flatten()[self.position_index()] = False

            in_terminal_state = torch.sum(self.map & self.terminal, dim=(1, 2)).byte()
            out_of_time = (self.step_counter >= self.max_steps)
            self.done = in_terminal_state | out_of_time

            return self.map.to(dtype=torch.float32, device=self.device), reward, self.done, self._reset, {}

    def lookahead(self):
        """
        Allows a one step lookahead from the current state
        :return: a tensor with dimensions ( num of simulations, action_taken, resulting state) and
        a tensor of rewards with dimensions( num of simulations, action_taken, reward)
        """
        with torch.no_grad():
            actions = torch.arange(self.action_space.n, device=self.device).repeat(self.n)
            x = self.position_x.repeat_interleave(self.action_space.n)
            y = self.position_y.repeat_interleave(self.action_space.n)

            zero_if_terminal = (~self.done).float().repeat(self.action_space.n)

            actions = one_hot(actions, 4).float() * zero_if_terminal.unsqueeze(1)
            actions = actions.matmul(self.t).long()
            x += actions[:, 0]
            y += actions[:, 1]
            x.clamp_(0, self.width - 1)
            y.clamp_(0, self.height - 1)

            map = torch.zeros(self.n * self.action_space.n, self.height, self.width, dtype=torch.long, requires_grad=False, device=self.device)

            base = torch.arange(self.n * self.action_space.n, device=self.device) * self.height * self.width
            offset = y * self.width + x
            index = base + offset

            map.flatten()[index] = 1.0

            reward = self.reward_present.float() * self.rewards
            reward = reward.unsqueeze(1).expand(self.n, self.action_space.n, -1, -1).flatten()[index]
            reward = reward * zero_if_terminal
            reward = reward.reshape(self.n, self.action_space.n).to(device=self.device)

            map = map.reshape(self.n, self.action_space.n, self.height, self.width)

            done = map & self.terminal
            done = torch.sum(done, dim=(2, 3)).byte()

            return map.to(dtype=torch.float32, device=self.device), reward, done, {}

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
                c = "{0:02.0f}".format(v * 100)
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
