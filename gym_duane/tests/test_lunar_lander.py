import gym
import gym_duane
import torch


def test_ll():
    device = 'cuda'
    env = gym.make('GridLunarLander-v0', n=2, device=device)
    obs = env.reset()

    #  0
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 15].item() == 1.0
    assert speed[0, 10].item() == 1.0

    # 1 = 0 + 1g
    actions = torch.tensor([4, 4], dtype=torch.long, device=device)
    obs, reward, done,info = env.step(actions)
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 16].item() == 1.0
    assert speed[0, 11].item() == 1.0

    # 2 = 1 + 1g
    actions = torch.tensor([4, 4], dtype=torch.long, device=device)
    obs, reward, done,info = env.step(actions)
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 18].item() == 1.0
    assert speed[0, 12].item() == 1.0

    # 0 = 2 + 1g -3t
    actions = torch.tensor([2, 2], dtype=torch.long, device=device)
    obs, reward, done, info = env.step(actions)
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 18].item() == 1.0
    assert speed[0, 10].item() == 1.0

    # -2 = 0 +1g -3t
    actions = torch.tensor([2, 2], dtype=torch.long, device=device)
    obs, reward, done, info = env.step(actions)
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 16].item() == 1.0
    assert speed[0, 8].item() == 1.0

    # -4 = 0 +1g -3t
    actions = torch.tensor([2, 2], dtype=torch.long, device=device)
    obs, reward, done, info = env.step(actions)
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 12].item() == 1.0
    assert speed[0, 6].item() == 1.0

    # -6 = 0 +1g -3t
    actions = torch.tensor([2, 2], dtype=torch.long, device=device)
    obs, reward, done, info = env.step(actions)
    map = obs.narrow(1, 0, 92)
    speed = obs.narrow(1, 92, 20)
    assert map[0, 6].item() == 1.0
    assert speed[0, 4].item() == 1.0
    assert done[0]