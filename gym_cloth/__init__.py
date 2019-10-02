from gym.envs.registration import register

register(
    id='cloth-v0',
    entry_point='gym_cloth.envs:ClothEnv',
)