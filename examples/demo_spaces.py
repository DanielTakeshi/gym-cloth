"""Similar to the random policy, but with better obs / action spaces.

Update Summer 2019 (starting in May): also using to test reward design.
Be careful about which actions are actually being used! Use this script to test
initializations for baselines, using the `demo_spaces` config. Then, set your
`demo_baselines` to be the same as the `demo_spaces` for training.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time
import yaml
import logging
import pickle
import datetime
from mpl_toolkits.mplot3d import Axes3D
from gym_cloth.envs import ClothEnv
from collections import defaultdict
from os.path import join
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)


def analytic(env, t, cfg):
    """Just put whatever hard coded rule I want, to test.

    Be careful about the action parameterization and if we clip or not.
    """
    if t==0:
        action = (-1, -1, 1, 1)
        #action = (-1, -1, 1, np.pi/4)
    elif t==1:
        action = (1, 1, -1, -1)
    elif t==2:
        action = (0, 0, 1, 0)
    elif t==3:
        action = (1, 0, -1, 0)
    else:
        action = env.get_random_action()
    f = 1
    action = (action[0], action[1], action[2]/f, action[3]/f)
    return action


def _corners_delta(env, t, cfg):
    pts = env.cloth.pts
    RAD_TO_DEG = 180 / np.pi

    # When computing angles, need to use x, y, not cx, cy.
    def _get_xy_vals(pt, targx, targy):
        x, y = pt.x, pt.y
        cx = (x - 0.5) * 2.0
        cy = (y - 0.5) * 2.0
        dx = targx - x
        dy = targy - y
        return (x, y, cx, cy, dx, dy)

    if t % 4 == 0:
        # Upper right.
        x, y, cx, cy, dx, dy = _get_xy_vals(pts[-1], targx=1, targy=1)
    elif t % 4 == 1:
        # Lower right.
        x, y, cx, cy, dx, dy = _get_xy_vals(pts[-25], targx=1, targy=0)
    elif t % 4 == 2:
        # Lower left.
        x, y, cx, cy, dx, dy = _get_xy_vals(pts[0], targx=0, targy=0)
    elif t % 4 == 3:
        # Upper left.
        x, y, cx, cy, dx, dy = _get_xy_vals(pts[24], targx=0, targy=1)

    if cfg['env']['clip_act_space']:
        action = (cx, cy, dx, dy)
    else:
        action = (x, y, dx, dy)
    return action


def _corners_nodelta(env, t, cfg):
    pts = env.cloth.pts
    RAD_TO_DEG = 180 / np.pi

    # When computing angles, need to use x, y, not cx, cy.
    def _get_xy_vals(pt):
        x, y = pt.x, pt.y
        cx = (x - 0.5) * 2.0
        cy = (y - 0.5) * 2.0
        return (x, y, cx, cy)

    if t % 4 == 0:
        # Upper right.
        x, y, cx, cy = _get_xy_vals(pts[-1])
        a = np.arctan2(1-y, 1-x)
        l = np.sqrt( (x-1)**2 + (y-1)**2 )
    elif t % 4 == 1:
        # Lower right.
        x, y, cx, cy = _get_xy_vals(pts[-25])
        a = np.arctan2(-y, 1-x)
        l = np.sqrt( (x-1)**2 + (y-0)**2 )
    elif t % 4 == 2:
        # Lower left.
        x, y, cx, cy = _get_xy_vals(pts[0])
        a = np.arctan2(-y, -x)
        l = np.sqrt( (x-0)**2 + (y-0)**2 )
    elif t % 4 == 3:
        # Upper left.
        x, y, cx, cy = _get_xy_vals(pts[24])
        a = np.arctan2(1-y, -x)
        l = np.sqrt( (x-0)**2 + (y-1)**2 )

    # Apply scaling factor to length if needed, since for non-delta actions,
    # length is just the fraction of the maximum number of pulls, which is
    # itself a tuned quantity.
    l = l * 1.0

    action = (x, y, l, a)
    if cfg['env']['clip_act_space']:
        action = (cx, cy, (l-0.5)*2, a/np.pi)
    else:
        action = (x, y, l, a)
    return action


def analytic_corners(env, t, cfg):
    """The analytic policy for pulling from corners.

    Use the np.arctan2(deltay, deltax) function for angle in [-pi, pi].
    Be careful about the action parameterization and if we clip or not.
    If you clip, you have to convert the x and y to each be in [0,1].
    """
    if cfg['env']['delta_actions']:
        return _corners_delta(env, t, cfg)
    else:
        return _corners_nodelta(env, t, cfg)


def run(config_file, render_path, file_path, result_path, load_state, max_episodes,
        random_pol=False):
    """Run a policy. Use this as the main testbed before running baselines-fork.
    """
    with open(config_file, 'r') as fh:
        cfg = yaml.safe_load(fh)
        seed = cfg['seed']
        stuff = '-clip_a-{}-delta_a-{}-obs-{}'.format(
                cfg['env']['clip_act_space'], cfg['env']['delta_actions'],
                cfg['env']['obs_type'])
        result_path = result_path.replace('.pkl', '{}.pkl'.format(stuff))

    # Save states into local directory, load from nfs diskstation.
    NFS = '/nfs/diskstation/seita/clothsim'
    state_path = join(file_path,"state_init.pkl")
    load_state_path = join(NFS,'state_init_med_49_coverage.pkl')
    num_episodes = 0
    stats_all = []

    # Should seed env this way, following gym conventions.
    if load_state:
        env = ClothEnv(config_file, start_state_path=load_state_path)
    else:
        env = ClothEnv(config_file)
    env.seed(seed)
    env.render(filepath=render_path)

    for ep in range(max_episodes):
        obs = env.reset()
        env.save_state(state_path)
        # Go through one episode and put information in `stats_ep`.
        # Put the first observation here to start.
        stats_ep = defaultdict(list)
        stats_ep['obs'].append(obs)
        done = False
        num_steps = 0

        while not done:
            if random_pol:
                #action = env.get_random_action(atype='touch_cloth')
                action = env.get_random_action(atype='over_xy_plane')
            else:
                #action = analytic(env, t=num_steps, cfg=cfg)
                action = analytic_corners(env, t=num_steps, cfg=cfg)

            # Apply the action.
            obs, rew, done, info = env.step(action)
            stats_ep['obs'].append(obs)
            stats_ep['rew'].append(rew)
            stats_ep['act'].append(action)
            stats_ep['done'].append(done)
            stats_ep['info'].append(info)
            num_steps += 1

        num_episodes += 1
        print("\nFinished {} episodes: {}\n".format(num_episodes, info))
        stats_all.append(stats_ep)

        # Just dump here to keep saving and overwriting.
        with open(result_path, 'wb') as fh:
            pickle.dump(stats_all, fh)

    assert len(stats_all) == max_episodes, len(stats_all)
    if env.render_proc is not None:
        env.render_proc.terminate()
        env.cloth.stop_render()


if __name__ == "__main__":
    # Use this to store results. For example, these can be used to save the
    # demonstrations that we later load to augment DeepRL training. We can
    # augment the file name later in `run()`.
    date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    result_pkl = 'demos-{}.pkl'.format(date)

    # Each time we use the environment, we need to pass in some configuration.
    file_path   = os.path.dirname(os.path.realpath(__file__))
    cfg_file    = join(file_path, '../cfg/demo_spaces.yaml')
    render_path = join(file_path, '../render/build')
    result_path = join(file_path, '../logs/{}'.format(result_pkl))

    # It's better if `load_state = False` if using the analytic policy. But
    # loading a state is much faster w.r.t. run time.
    random_pol    = False
    load_state    = False
    max_episodes  = 10

    run(cfg_file, render_path, file_path, result_path, load_state, max_episodes,
        random_pol)
