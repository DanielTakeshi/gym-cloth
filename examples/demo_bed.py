"""For demonstrations with a bed.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time
import yaml
from mpl_toolkits.mplot3d import Axes3D
from gym_cloth.envs import ClothEnv
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)


def test_init(config_file):
    """Test environment initialization, and then hard-coded actions.

    For rendering, use the environment's `env.render()` method.
    """
    load_state = True

    env = ClothEnv(config_file)
    print("\nObs space: {}".format(env.observation_space))
    print("Act space: {}".format(env.action_space))

    # Start the environment and render it.
    this_dir = os.path.dirname(os.path.realpath(__file__))
    path_to_renderer = os.path.join(this_dir, "../render/build")
    env.render(filepath=path_to_renderer)
    start = time.time()

    # Save or load the state -- but use a FULL path. Otherwise it annoyingly
    # saves in a different directory, must be a cython thing?
    state_path = os.path.join(this_dir,"test_saving_state_bed.pkl")
    if load_state:
        obs = env.load_state(cloth_file=state_path)
    else:
        obs = env.reset()
        env.save_state(state_path)

    print("reset took {}".format(time.time() - start))
    print("\nJust reset, obs is:\n{}\nshape: {}".format(obs, obs.shape))

    # Do your hard-coded actions here.

    p0 = env.cloth.pts[-1]
    action = ((p0.x, p0.y), 0.60, 2)
    obs, rew, done, info = env.step(action)
    print("demo_bed.py, done: {}".format(done))

    p1 = env.cloth.pts[-25]
    action = ((p1.x, p1.y), 0.50, 7)
    obs, rew, done, info = env.step(action)
    print("demo_bed.py, done: {}".format(done))

    p2 = env.cloth.pts[-25]
    action = ((p2.x, p2.y), 0.50, 3)
    obs, rew, done, info = env.step(action)
    print("demo_bed.py, done: {}".format(done))

    p3 = env.cloth.pts[-25]
    action = ((p3.x, p3.y), 0.30, 2)
    obs, rew, done, info = env.step(action)
    print("demo_bed.py, done: {}".format(done))

    p4 = env.cloth.pts[-(25*3)-1]
    action = ((p4.x, p4.y), 0.30, 1)
    obs, rew, done, info = env.step(action)
    print("demo_bed.py, done: {}".format(done))

    # Kill the render process. Normally it's invoked when a state is done but
    # we can also hard-code it like this.
    print("\nstopping the render proc and renderer")
    env.render_proc.terminate()
    env.cloth.stop_render()


if __name__ == "__main__":
    # Each time we use the environment, we need to pass in some configuration.
    file_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(file_path, '../cfg/demo_bed.yaml')
    
    with open(cfg_file, 'r') as fh:
        cfg = yaml.load(fh)
        np.random.seed(cfg['seed'])

    test_init(cfg_file)
