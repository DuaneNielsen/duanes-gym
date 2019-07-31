import pickle

import numpy as np

import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import torch


class LookAhead(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_tuple = None

    def reset(self):
        o = self.env.reset()
        self.step_tuple = (o, 0.0, False, {})
        return o

    def step(self, action):
        self.step_tuple = self.env.step(action)
        return self.step_tuple

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def lookahead(self):

        obs = []
        reward = []
        done = []
        info = []

        # if we are in done state, just return the terminal state with no reward
        if self.step_tuple[2]:
            obs = [self.step_tuple[0]]*self.action_space.n
            reward = [self.step_tuple[1]]*self.action_space.n
            done = [self.step_tuple[2]]*self.action_space.n
            info = [self.step_tuple[3]]*self.action_space.n

        # otherwise, look at every action and tell me what's going to happen
        else:
            anchor = pickle.dumps(self.env)
            for i, action in enumerate(range(self.action_space.n)):
                next = pickle.loads(anchor)
                o, r, d, i = next.step(action)
                obs.append(o)
                reward.append(r)
                done.append(d)
                info.append(i)

        obs = np.array(obs)
        reward = np.array(reward)
        done = np.array(done)

        return obs, reward, done, info


class Reset(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.done = False

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.done:
            self.done = False
            o = self.env.reset()
            return o, 0.0, False, True, {}
        o, r, d, i = self.env.step(action)
        self.done = d
        return o, r, d, False, i

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def lookahead(self):
        return self.env.lookahead()


class BatchTensor(gym.Wrapper):
    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self.device = device

    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).unsqueeze(0)

    def step(self, action):
        action = action.item()
        o, r, d, rst, i = self.env.step(action)
        obs = torch.from_numpy(o).unsqueeze(0).to(self.device)
        reward = torch.tensor([r], dtype=torch.float32, device=self.device)
        done = torch.tensor([d], dtype=torch.uint8, device=self.device)
        reset = torch.tensor([rst], dtype=torch.uint8, device=self.device)
        return obs, reward, done, reset, i

    def lookahead(self):
        o, r, d, i = self.env.lookahead()
        obs = torch.from_numpy(o).unsqueeze(0).to(self.device)
        reward = torch.tensor([r], dtype=torch.float32, device=self.device)
        done = torch.tensor([d], dtype=torch.uint8, device=self.device)
        return obs, reward, done, i

    def render(self, **kwargs):
        return self.env.render(**kwargs)
