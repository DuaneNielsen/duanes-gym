import gym
from random import randint
import envs
import numpy as np
import torch
from wrappers import LookAhead, Reset, BatchTensor

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    env = gym.make('SimpleGrid-v2', n=3, device='cpu', map_string="""
    [
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, S, E, E],
    [E, E, E, E, T(1.0)]
    ]
    """)

    assert env.observation_space_shape[0][0] == 4
    assert env.observation_space_shape[0][1] == 5

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


def test_simple_grid_v2_render():
    env = gym.make('SimpleGrid-v2', n=4000, device='cpu', map_string="""
    [
    [E, E, E, E, E],
    [E, E, E, E, E],
    [E, E, S, E, E],
    [E, E, E, E, T(1.0)]
    ]
    """)

    print('')
    obs = env.reset()
    env.render()
    print('')

    action = torch.randint(4, size=(4000,))
    obs = env.step(action)
    env.render()
    print('')

    action = torch.randint(4, size=(4000,))
    obs = env.step(action)
    env.render()
    print('')

    action = torch.randint(4, size=(4000,))
    obs = env.step(action)
    env.render()
    print('')

    action = torch.randint(4, size=(4000,))
    obs = env.step(action)
    env.render()
    print('')


def test_bandit():
    env = gym.make('SimpleGrid-v3', n=10, device='cuda', max_steps=50,
                   map_string='[[T(-1.0), S, T(1.0)]]')
    obs = env.reset()
    env.render()

    for i in range(10):
        action = torch.LongTensor(10).random_(0, 2).cuda()
        print(action)
        obs, reward, done, info, reset = env.step(action)
        print(reward, done)
        env.render()


def test_lava():
    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [S, L, T]
        ]
        """)

    env.reset()
    state, reward, done, info = env.step(torch.tensor([1]))
    assert reward[0].item() == -1.0
    assert done[0] == 1


def test_simple_grid_line():
    env = gym.make('SimpleGrid-v2', n=3, device='cpu', map_string="""
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
    assert torch.all(done)

    action = torch.LongTensor([0, 0, 0])
    obs, reward, done, info = env.step(action)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    init = init.repeat(3, 1, 1)
    assert torch.allclose(expected, obs)


def test_simple_grid_line_reset():
    env = gym.make('SimpleGrid-v2', n=2, device='cpu', map_string="""
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


def test_start():
    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [S, E],
        [E, E]
        ]
        """)
    state = env.reset()

    expected = torch.zeros(2, 2)
    expected[0, 0] = 1.0

    assert torch.allclose(expected, state)

    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [E, E],
        [E, S]
        ]
        """)
    state = env.reset()

    expected = torch.zeros(2, 2)
    expected[1, 1] = 1.0

    assert torch.allclose(expected, state)

    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [E, S],
        [E, E]
        ]
        """)
    state = env.reset()

    expected = torch.zeros(2, 2)
    expected[0, 1] = 1.0
    print(expected, state)

    assert torch.allclose(expected, state)

    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [E, E],
        [S, E]
        ]
        """)
    state = env.reset()

    expected = torch.zeros(2, 2)
    expected[1, 0] = 1.0
    print(expected, state)

    assert torch.allclose(expected, state)


def test_simple_grid_y():
    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [S]
        ]
        """)
    env.reset()
    env.step(torch.tensor([0]))

    env = gym.make('SimpleGrid-v2', n=1, device='cpu', map_string="""
        [
        [T, S, T]
        ]
        """)
    env.reset()
    env.step(torch.tensor([1]))
    env.step(torch.tensor([1]))


def test_done_flag():
    env = gym.make('SimpleGrid-v3', n=3, device='cuda', max_steps=60, map_string="""
        [
        [T(-1.0), S, T(1.0)]
        ]
        """)

    s = env.reset()
    for _ in range(90):
        action = torch.randint(4, (3,)).cuda()
        n, reward, done, reset, info = env.step(action)
        term = torch.sum(n * torch.tensor([1.0, 0.0, 1.0]).cuda(), dim=[1, 2])
        assert torch.allclose(term, done.float())


def test_reset_flag():
    env = gym.make('SimpleGrid-v3', n=3, device='cuda', max_steps=60, map_string="""
        [
        [T(-1.0), S, T(1.0)]
        ]
        """)

    s = env.reset()
    for _ in range(90):
        action = torch.randint(4, (3,)).cuda()
        n, reward, done, reset, info = env.step(action)
        term = torch.sum(s * torch.tensor([1.0, 0.0, 1.0]).cuda(), dim=[1, 2])
        assert torch.allclose(term, reset.float())
        s = n.clone()


def test_terminal_states():
    env = gym.make('SimpleGrid-v3', n=3, device='cuda', max_steps=40, map_string="""
        [
        [T(-1.0), S, T(1.0)]
        ]
        """)

    s = env.reset()
    for _ in range(100):
        action = torch.randint(4, (3,)).cuda()
        n, reward, done, reset, info = env.step(action)

        term = torch.sum(n * torch.tensor([1.0, 0.0, 1.0]).cuda(), dim=[1, 2])
        assert torch.allclose(term, done.float())

        left = torch.tensor([[1.0, 0.0, 0.0]]).cuda()
        right = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        left_action = (action == 0) & ~reset
        right_action = (action == 1) & ~reset
        assert torch.allclose(n[left_action], left)
        assert torch.allclose(n[right_action], right)
        s = n.clone()

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


def test_cuda():
    env = gym.make('SimpleGrid-v2', n=10000, map_string='[[S, E, E, E, E, E, E, E, E, E, T(1.0)]]', device='cuda')
    obs = env.reset()
    env.render()

    for i in range(500):
        action = torch.LongTensor(10000).random_(2).to('cuda')
        obs, reward, done, info = env.step(action)
        env.render()


def test_reset_rewards():
    env = gym.make('SimpleGrid-v3', n=2, device='cpu', map_string='[[S, E(1.0), E, T]]', max_steps=10)

    env.reset()

    action = torch.tensor([1, 1])
    state, reward, done, reset, info = env.step(action)
    assert reward[0] == 1.0
    assert reward[1] == 1.0

    action = torch.tensor([1, 1])
    state, reward, done, reset, info = env.step(action)
    assert reward[0] == 0.0
    assert reward[1] == 0.0

    action = torch.tensor([0, 0])
    state, reward, done, reset, info = env.step(action)
    assert reward[0] == 0.0

    action = torch.tensor([0, 0])
    state, reward, done, reset, info = env.step(action)
    assert reward[0] == 0.0




def test_next_states():
    env = gym.make('SimpleGrid-v3', n=2, device='cpu', map_string='[[T(1.0), S, T(-1.0)]]', max_steps=10)

    env.reset()
    env.render()
    states, rewards, done, info = env.lookahead()
    expected_state = torch.tensor([
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]]
        ],
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]]
        ]
    ])
    assert torch.allclose(states, expected_state)

    exp_reward = torch.tensor([
        [
            [1.0, -1.0, 0.0, 0.0]
        ],
        [
            [1.0, -1.0, 0.0, 0.0]
        ]
    ])
    assert torch.allclose(rewards, exp_reward)

    action = torch.tensor([0, 1])

    state, reward, done, reset, info = env.step(action)

    env.render()
    states, rewards, done, info  = env.lookahead()
    expected_state = torch.tensor([
        [
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]]
        ],
        [
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0]]
        ]
    ])
    assert torch.allclose(states, expected_state)

    exp_reward = torch.tensor([
        [
            [0.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0, 0.0]
        ]
    ])
    assert torch.allclose(rewards, exp_reward)

    action = torch.tensor([0, 1])

    state, reward, done, reset, info = env.step(action)
    print(state, reward)
    next_state, next_reward, done, info = env.lookahead()
    print(next_state, next_reward)

    expected_state = torch.tensor([
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]]
        ],
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]]
        ]
    ])
    assert torch.allclose(next_state, expected_state)

    exp_reward = torch.tensor([
        [
            [1.0, -1.0, 0.0, 0.0]
        ],
        [
            [1.0, -1.0, 0.0, 0.0]
        ]
    ])
    assert torch.allclose(next_reward, exp_reward)



def test_step_counter():
    env = gym.make('SimpleGrid-v3', n=40, device='cuda', map_string='[[S, E, E, E, E, T(1.0)]]', max_steps=5)

    env.reset()
    for _ in range(500):
        action = torch.randint(0, 3, (40,)).cuda()
        #action = torch.tensor([0, 1]).cuda()
        state, reward, done, reset, info = env.step(action)
        env.render()
        #assert reward[0] == 0.0


def test_lookahead_wrapper():
    env = gym.make('LunarLander-v2')
    env = LookAhead(env)
    obs = env.reset()
    done = False

    for _ in range(1000):
        if done:
            obs = env.reset()
        obs, reward, done, info = env.lookahead()
        assert obs.shape[0] == 4
        action = randint(0, env.action_space.n -1)
        obs, reward, done, info = env.step(action)
        env.render()


def test_reset_wrapper():
    env = gym.make('LunarLander-v2')
    env = Reset(env)
    obs = env.reset()
    total_reward = 0.0

    for _ in range(1000):
        action = randint(0, env.action_space.n -1)
        obs, reward, done, reset, info = env.step(action)
        total_reward += reward
        if reset: print('RESET')
        env.render()

    assert total_reward < 0.0


def test_batch_tensor_wrapper():
    env = gym.make('LunarLander-v2')
    env = Reset(env)
    env = BatchTensor(env, device='cuda')
    obs = env.reset()
    total_reward = 0.0

    for _ in range(1000):
        action = torch.tensor([randint(0, env.action_space.n -1)]).unsqueeze(0)
        obs, reward, done, reset, info = env.step(action)
        env.render()
        total_reward += reward

    assert total_reward < 0.0

def test_batch_tensor_wrapper_with_lookahead():
    env = gym.make('LunarLander-v2')
    env = LookAhead(env)
    env = Reset(env)
    env = BatchTensor(env, device='cuda')
    obs = env.reset()
    total_reward = 0.0

    for _ in range(1000):
        obs, reward, done, info = env.lookahead()
        action = torch.tensor([randint(0, env.action_space.n -1)]).unsqueeze(0)
        obs, reward, done, reset, info = env.step(action)
        env.render()
        total_reward += reward.item()

    assert total_reward < 0.0
    print(total_reward)