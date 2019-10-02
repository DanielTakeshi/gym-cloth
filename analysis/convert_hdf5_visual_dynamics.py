"""Convert to data for visual dynamics models.

Edit: not complete yet.
"""
import os
import cv2
import sys
import pickle
import numpy as np
import h5py
np.set_printoptions(suppress=True, edgeitems=2000, linewidth=200)
from os.path import join


def get_numpy(data):
    """Assume we clip the actions here.
    """
    all_images = []
    all_actions = []

    for eidx,ep in enumerate(data):
        images = np.array(ep['obs'])
        actions = np.array(ep['act'])
        assert len(images.shape) == 4 and images.shape[1:] == (200,200,3), images.shape
        assert len(actions.shape) == 2 and actions.shape[-1] == 4, actions.shape
        all_images.append(images)
        all_actions.append(actions)

    # We can't turn to numpy due to different episode lengths.
    print('  len images: {}'.format(len(all_images)))
    print('  len actions: {}'.format(len(all_actions)))
    return (all_images, all_actions)


if __name__ == "__main__":
    """Note that the demonstration data are generated via:

        python examples/analytic.py oracle --max_episodes=500 --seed=1337

    with appropriate cfg changes + random seed. I then merge lists together
    manually if desired.
    """
    if not os.path.exists('data'):
        os.makedirs('data')

    tier1_f = 'logs/demos-2019-08-02-15-48-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_1000.pkl'
    tier2_f = 'logs/demos-2019-08-04-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    tier3_f = 'logs/demos-2019-08-03-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'
    with open(tier1_f, 'rb') as fh:
        data_t1 = pickle.load(fh)
    with open(tier2_f, 'rb') as fh:
        data_t2 = pickle.load(fh)
    with open(tier3_f, 'rb') as fh:
        data_t3 = pickle.load(fh)
    print('loaded tier1 data, num episodes {}'.format(len(data_t1)))
    print('loaded tier2 data, num episodes {}'.format(len(data_t2)))
    print('loaded tier3 data, num episodes {}'.format(len(data_t3)))

    combo = data_t1 + data_t2 + data_t3
    print('Creating tier1 np data.')
    d1_img, d1_act = get_numpy(data_t1)
    print('Creating tier2 np data.')
    d2_img, d2_act = get_numpy(data_t2)
    print('Creating tier3 np data.')
    d3_img, d3_act = get_numpy(data_t3)
    print('Creating combo np data.')
    dc_img, dc_act = get_numpy(combo)

    h5f_01    = h5py.File('tier01_data.h5', 'w')
    h5f_02    = h5py.File('tier02_data.h5', 'w')
    h5f_03    = h5py.File('tier03_data.h5', 'w')
    h5f_combo = h5py.File('tier_combo_data.h5', 'w')

    h5f_01.create_dataset('data_tier01', data=d1_img)
    h5f_02.create_dataset('data_tier02', data=d2_img)
    h5f_03.create_dataset('data_tier03', data=d3_img)
    h5f_combo.create_dataset('data_combo', data=dc_img)
