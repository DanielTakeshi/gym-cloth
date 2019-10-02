"""Use this script to test randomly initializing the environment.

Ideally, we can visualize the distribution of starting states, such as by
saving a bunch of images or short GIFs and scrolling through them. We probably
need this in an appendix somewhere in the paper.

Some useful refs for understanding gym environments:

  https://github.com/openai/gym/blob/master/gym/envs/
  https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
  https://github.com/openai/gym-soccer
  https://github.com/jeffmahler/dex-net/tree/master/cfg (private, yaml)
  https://github.com/jeffmahler/dex-net/tree/master/src/dexnet/envs (private, envs)

We are going to pass in a configuration file, so we're not exactly following
the gym interface, but the API should match. See examples/analytic.py.
"""
import numpy as np
import argparse
import os
import sys
import time
import yaml
from gym_cloth.envs import ClothEnv
from os.path import join
from collections import defaultdict
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)

# Here's how one can profile, if we wanted to.
#import pstats, cProfile
#cProfile.runctx('test_init(env)', globals(), locals(), 'main.prof')
#s = pstats.Stats("main.prof")
#s.strip_dirs().sort_stats("time").print_stats(25)
#s.strip_dirs().sort_stats("cumtime").print_stats(25)


# I copied the relevant parts from examples/analytic.py.
if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument("--num_obs", type=int, default=10)
    args = pp.parse_args()
    args.file_path = fp = os.path.dirname(os.path.realpath(__file__))
    args.cfg_file = join(fp, '../cfg/demo_baselines.yaml') # BASELINES!
    args.render_path = join(fp, '../render/build')    # Must be compiled!

    with open(args.cfg_file, 'r') as fh:
        cfg = yaml.safe_load(fh)
        seed = cfg['seed']
        print('\nseed:           {}'.format(seed))
        print('clip_act_space: {}'.format(cfg['env']['clip_act_space']))
        print('delta_actions:  {}'.format(cfg['env']['delta_actions']))
        print('obs_type:       {}'.format(cfg['env']['obs_type']))

    # Should seed env this way, following gym conventions.
    env = ClothEnv(args.cfg_file)
    env.seed(seed)
    env.render(filepath=args.render_path)

    # Just collect a bunch of observations.
    stats = defaultdict(list)
    for nb_obs in range(args.num_obs):
        obs = env.reset()
        env.cloth.stop_render()
        stats['coverage'].append(env._compute_coverage())
        stats['variance'].append(env._compute_variance())
        print("\nNow collected {} observations".format(nb_obs+1))
        assert nb_obs+1 == len(stats['coverage'])
        print('  coverage: {:.2f} +/- {:.1f}'.format(np.mean(stats['coverage']),
                                                     np.std(stats['coverage'])))
        print('  variance: {:.2f} +/- {:.1f}'.format(np.mean(stats['variance']),
                                                     np.std(stats['variance'])))
        print('(remember, variance is really _inverse_ variance)')

    if env.render_proc is not None:
        env.render_proc.terminate()
        env.cloth.stop_render()
