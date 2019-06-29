import gym
from random import randint
import envs
import numpy as np
import torch


def test_simple_grid():
    env = gym.make('SimpleGrid-v0', n=10, map_string='[T(-1.0), S, T(1.0)]')
    obs = env.reset()
    env.render()

    for i in range(10):
        action = torch.LongTensor(10).random_(-1, 2)
        print(action)
        obs, reward, done, info = env.step(action)
        print(reward, done)
        env.render()


def test_line_grid():
    env = gym.make('SimpleGrid-v0', n=10, map_string='[S, E, E, E, E, E, E, E, E, E, T(1.0)]')
    obs = env.reset()
    env.render()

    for i in range(200):
        action = torch.LongTensor(10).random_(-1, 2)
        print(action)
        obs, reward, done, info = env.step(action)
        print(reward, done)
        env.render()


def test_simple_grid_v2():
    env = gym.make('SimpleGrid-v2', n=3, map_string="""
    [
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, S, E, E],
    [E, E, E, E, T(1.0)]
    ]
    """)

    obs = env.reset()

    init = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(init, obs)

    action = torch.LongTensor([1, 1, 1])
    obs, reward, done, info = env.step(action)

    step1 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    )

    step1 = step1.repeat(3, 1, 1)
    assert torch.allclose(step1, obs)

    action = torch.LongTensor([1, 1, 1])
    obs, reward, done, info = env.step(action)
    action = torch.LongTensor([3, 3, 3])
    obs, reward, done, info = env.step(action)

    step2 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    )

    step2 = step2.repeat(3, 1, 1)
    assert torch.allclose(step2, obs)
    assert torch.allclose(reward, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(done, torch.ByteTensor([1, 1, 1]))

    obs = env.reset()

    init = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(init, obs)

    action = torch.LongTensor([0, 0, 0])
    obs, reward, done, info = env.step(action)

    step1 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    )
    assert torch.allclose(step1, obs)
    assert torch.allclose(reward, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(done, torch.ByteTensor([0, 0, 0]))

    # assert torch.allclose(env.terminated, torch.ByteTensor([0, 0, 0]))

    for i in range(100):
        action = torch.LongTensor(3).random_(0, 3)
        obs, reward, done, info = env.step(action)


def test_simple_grid_line():
    env = gym.make('SimpleGrid-v2', n=3, map_string="""
        [
        [S, E, E, T]
        ]
        """)

    obs = env.reset()

    init = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(init, obs)

    action = torch.LongTensor([0, 0, 0])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)

    action = torch.LongTensor([2, 2, 2])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)

    action = torch.LongTensor([1, 1, 1])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)

    action = torch.LongTensor([1, 1, 1])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)

    action = torch.LongTensor([1, 1, 1])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)

    action = torch.LongTensor([0, 0, 0])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)


def test_simple_grid_line_reset():
    env = gym.make('SimpleGrid-v2', n=2, map_string="""
        [
        [S, E, E, T]
        ]
        """)

    obs = env.reset()

    init = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    ])

    assert torch.allclose(init, obs)

    action = torch.LongTensor([1, 0])
    obs, reward, done, info = env.step(action)
    obs, reward, done, info = env.step(action)
    obs, reward, done, info = env.step(action)

    expected = torch.tensor([
        [
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    ])
    expected_done = torch.ByteTensor([1, 0])

    assert torch.allclose(expected, obs)
    assert torch.allclose(expected_done, done)

    obs, reward, done, info = env.step(action)

    expected = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    ])
    expected_done = torch.ByteTensor([0, 0])

    assert torch.allclose(expected, obs)
    assert torch.allclose(expected_done, done)


def test_parse():
    from lark import Lark

    # % import common.STRING // import from terminal library

    l = Lark(
        '''
        start: "[" row "]"
        row: token ("," token)*
        token: WORD ( "(" SIGNED_NUMBER ")" )?
        
        %import common.WORD
        %import common.SIGNED_NUMBER
        %ignore " "           // Disregard spaces in text
        ''')

    class Grid:
        def __init__(self):
            self.row = 0
            self.start = 0
            self.terminal = []
            self.rewards = []

        def element(self, *args):
            reward = 0.0 if len(args) == 1 else float(args[1].value)
            terminal_state = 1 if args[0] == 'T' else 0
            if args[0] == 'S':
                self.start = self.row
            self.rewards.append(reward)
            self.terminal.append(terminal_state)
            self.row += 1

    g = Grid()
    print(l.parse('[T]'))
    print(l.parse('[T(-1.0), S, T(1.0)]'))
    tree = l.parse('[T(-1.0), S, T(1.0)]')
    tree = l.parse('[T(-1.0), S, T(1.0)]')
    for token in tree.children[0].children:
        print(*token.children)
        g.element(*token.children)

    print(g.start)
    print(torch.tensor(g.terminal))
    print(torch.tensor(g.rewards))


def test_parse_column():
    from lark import Lark

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

    class Grid:
        def __init__(self):
            self.row = 0
            self.start = 0
            self.terminal = []
            self.rewards = []

        def element(self, *args):
            reward = 0.0 if len(args) == 1 else float(args[1].value)
            terminal_state = 1 if args[0] == 'T' else 0
            if args[0] == 'S':
                self.start = self.row
            self.rewards.append(reward)
            self.terminal.append(terminal_state)
            self.row += 1

    g = Grid()

    map_str = """
    [
    [S, E, E, E, E],
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, E, E, T(1.0)]
    ]
    """

    # tree = l.parse('[[T(-1.0), S, T(1.0)], [T(-1.0), S, T(1.0)]]')
    str = map_str.strip("\n")
    tree = l.parse(str)
    height = len(tree.children)
    width = len(tree.children[0].children)
    print(height)
    print(width)

    terminal_states = torch.zeros(height, width)
    rewards = torch.zeros(height, width)
    start_pos = (0, 0)

    def elem(*args):
        reward = 0.0 if len(args) == 1 else float(args[1].value)
        terminal_state = 1 if args[0] == 'T' else 0
        is_start = args[0] == 'S'
        return reward, terminal_state, is_start

    for i, row in enumerate(tree.children):
        for j, token in enumerate(row.children):
            reward, terminal_state, is_start = elem(*token.children)
            terminal_states[i, j] = terminal_state
            rewards[i, j] = reward
            if is_start:
                start_pos = (i, j)

    print(terminal_states)
    print(rewards)
    print(start_pos)
