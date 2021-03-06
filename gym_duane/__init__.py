import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Banana-v0',
    entry_point='gym_duane.envs:BananaEnv',
)

register(
    id='PymunkPong-v0',
    entry_point='gym_duane.envs:PongEnv',
)

register(
    id='AlphaRacer2D-v0',
    entry_point='gym_duane.envs:AlphaRacer2DEnv'
)

register(
    id='Bouncer-v0',
    entry_point='gym_duane.envs:BounceEnv'
)

register(
    id='SimpleGrid-v0',
    entry_point='gym_duane.envs:SimpleGrid'
)

register(
    id='SimpleGrid-v2',
    entry_point='gym_duane.envs:SimpleGridV2'
)

register(
    id='SimpleGrid-v3',
    entry_point='gym_duane.envs:SimpleGridV3'
)

register(
    id='GridLunarLander-v0',
    entry_point='gym_duane.envs:LunarLander'
)