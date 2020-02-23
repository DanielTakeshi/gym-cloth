"""Inspect data that was generated from demo policy, to filter for DDPG and
report demonstrator data for paper.

We have max length-10 episodes. For shorter episodes, it's because of pulling
out of bounds (no corner visible) so we should ignore those cases.  For others,
there are a few cases when all the 10 time steps are needed to get high
coverage, but mostly, it is probably fine to ignore the last 5. Except for when
we get high coverage, which might be interesting ... say if any reward in the
last 5 get more than 0.1 delta coverage, we include it? That might be something
extra we can do.

August 3, 2019: actually I think it's best if we simply exit with a success
(huge reward bonus). Then shorter episodes can be good. If not, then the
previous paragraph applies.

Sept 2019: average depth channel values of the 100x100 resized images:

Tier 1 Depth Demonstrator:
  chan. 1: 134.1 +/- 77.8
  chan. 2: 134.1 +/- 77.8
  chan. 3: 134.1 +/- 77.8

Tier 2 Depth Demonstrator:
  chan. 1: 134.2 +/- 75.7
  chan. 2: 134.2 +/- 75.7
  chan. 3: 134.2 +/- 75.7

Tier 3 Depth Demonstrator:
  chan. 1: 143.5 +/- 72.4
  chan. 2: 143.5 +/- 72.4
  chan. 3: 143.5 +/- 72.4

Tier 1 Color Demonstrator:
  chan. 1: 158.9 +/- 68.8
  chan. 2: 109.2 +/- 32.1
  chan. 3: 78.4 +/- 13.5

Tier 2 Color Demonstrator:
  chan. 1: 157.3 +/- 66.1
  chan. 2: 113.5 +/- 35.0
  chan. 3: 85.8 +/- 19.0

Tier 3 Color Demonstrator:
  chan. 1: 153.5 +/- 65.7
  chan. 2: 115.4 +/- 37.7
  chan. 3: 91.0 +/- 24.1
"""
import os
import cv2
import sys
import pickle
import numpy as np
np.set_printoptions(suppress=True, edgeitems=2000, linewidth=200)
from os.path import join


def analyze(data, lengths, use_img=False):
    """Will save a whole bunch of data if use_img=True, so be careful.
    """
    num_sparse_success = 0
    num_oob = 0
    num_tear = 0
    start_coverage_rew = []
    start_variance_rew = []
    end_coverage_rew = []
    end_variance_rew = []
    actions = []
    num_transitions = 0

    # For average pixel values?
    average_img = np.zeros((100,100,4))
    average_nb = 0

    for eidx,ep in enumerate(data):
        # We can use this to get coverage per-time step
        # Hacky but we do what we have to do ... assumes we know a specified index.
        # I use this for generating examples of oracle corner pulling policy.
        if eidx == 4:
            print('\nat our desired index, here is the e info, for the last time stpe:')
            _einfo = ep['info']
            #print(_einfo)
            for item in _einfo:
                print(item)
            print()

        eplen = len(ep['rew'])
        num_transitions += len(ep['rew'])
        for t in range(eplen):
            img = ep['obs'][t]
            act = ep['act'][t]
            rew = ep['rew'][t]
            fin = ep['done'][t]
            # Ah, can use this for actual coverage.
            suffix = 'img_ep_{}_t_{}_rew_{:.1f}_act_{:.2f}_{:.2f}_{:.2f}_{:.2f}.png'.format(
                    str(eidx).zfill(3), str(t).zfill(2), rew, act[0], act[1], act[2], act[3])
            fname = join('logs',suffix)

            if use_img:
                # Right now images are actually 224x224 and we forcibly resize in code,
                # because this gives the option of using pre-trained residual networks.
                assert img is not None
                assert img.shape == (200,200,3) or img.shape == (224,224,3) \
                        or img.shape == (224,224,4), img.shape
                cv2.imwrite(fname, img)

                # JUST IN CASE. The resized one is what the network sees.
                coverage = 100.0
                x = cv2.resize(img, (100,100))
                suffix = 'img_resized_ep_{}_t_{}_rew_{:.1f}_act_{:.2f}_{:.2f}_{:.2f}_{:.2f}_c_cov{:.1f}.png'.format(
                        str(eidx).zfill(3), str(t).zfill(2), rew, act[0], act[1], act[2], act[3], coverage)
                fname = join('logs',suffix)
                cv2.imwrite(fname, x[:,:,:3])
                suffix = 'img_resized_ep_{}_t_{}_rew_{:.1f}_act_{:.2f}_{:.2f}_{:.2f}_{:.2f}_d_cov{:.1f}.png'.format(
                        str(eidx).zfill(3), str(t).zfill(2), rew, act[0], act[1], act[2], act[3], coverage)
                fname = join('logs',suffix)
                cv2.imwrite(fname, x[:,:,3])
                average_img += x
                average_nb += 1

            # Be careful, actions may be outside range; we clip during env steps.
            actions.append(act[0])
            actions.append(act[1])
            actions.append(act[2])
            actions.append(act[3])

        if use_img:
            # For very last image.
            img = ep['obs'][eplen]
            suffix = 'img_ep_{}_t_{}_lastobs.png'.format(str(eidx).zfill(3), str(eplen).zfill(2))
            fname = join('logs',suffix)
            cv2.imwrite(fname, img)

            # JUST IN CASE.
            x = cv2.resize(img, (100,100))
            suffix = 'img_resized_ep_{}_t_{}_lastobs_c.png'.format(
                    str(eidx).zfill(3), str(eplen).zfill(2))
            fname = join('logs',suffix)
            cv2.imwrite(fname, x[:,:,:3])
            suffix = 'img_resized_ep_{}_t_{}_lastobs_d.png'.format(
                    str(eidx).zfill(3), str(eplen).zfill(2))
            fname = join('logs',suffix)
            cv2.imwrite(fname, x[:,:,3])
            average_img += x
            average_nb += 1

        # The last information, note `num_steps` is one less than num obs stored.
        info_last = ep['info'][-1]
        if info_last['have_tear']:
            num_tear += 1
        if info_last['out_of_bounds']:
            num_oob += 1

        # Don't forget to check rewards. We may filter?
        # The 'start' statistics should match across all ep['info'] items.
        start_coverage_rew.append( info_last['start_coverage'] )
        start_variance_rew.append( info_last['start_variance_inv'] )
        end_coverage_rew.append( info_last['actual_coverage'] )
        end_variance_rew.append( info_last['variance_inv'] )

        if eidx % 100 == 0:
            print('done with episode {}'.format(eidx))
        if sum(ep['rew']) == 1:
            num_sparse_success += 1

    # Copy and paste these for reporting results.
    print('num sparse success: {} out of {}'.format(num_sparse_success, len(data)))
    print('(note: ignore this if we are not using sparse rewards)\n')
    print('loaded data, length (i.e., episodes) {}'.format(len(data)))
    print('episode lengths (wrt t): max / min / mean: {:.1f} / {:.1f} / {:.2f} +/- {:.1f}'.format(
            np.max(lengths), np.min(lengths), np.mean(lengths), np.std(lengths)))
    print('num_oob: {}'.format(num_oob))
    print('num_tear: {}'.format(num_tear))

    # Informally, the 'success' threshold is 0.92 for end-of-episode.
    thresh = 0.90
    num_below = np.sum(np.array(end_coverage_rew) < thresh)
    print('num coverage below {}: {}'.format(thresh, num_below))
    print('total transitions: {}'.format(num_transitions))

    # Actually, std should prob be at % level not decimals ?
    start_coverage_rew = np.array(start_coverage_rew) * 100
    end_coverage_rew = np.array(end_coverage_rew) * 100

    print('coverage at start: {:.1f} +/- {:.1f}'.format(
            np.mean(start_coverage_rew), np.std(start_coverage_rew)))
    print('coverage at end:   {:.1f} +/- {:.1f}'.format(
            np.mean(end_coverage_rew), np.std(end_coverage_rew)))
    print('inv-var at start:  {:.3f} +/- {:.1f}'.format(
            np.mean(start_variance_rew), np.std(start_variance_rew)))
    print('inv-var at end:    {:.3f} +/- {:.1f}'.format(
            np.mean(end_variance_rew), np.std(end_variance_rew)))
    print('actions, len {}, max {:.3f}, min {:.3f}, mean {:.3f}'.format(
            len(actions), np.max(actions), np.min(actions), np.mean(actions)))

    # Average img value:
    average_img = average_img / average_nb
    print('\naverage values across three channels')
    print('  chan. 1: {:.1f} +/- {:.1f}'.format(np.mean(average_img[:,:,0]), np.std(average_img[:,:,0])))
    print('  chan. 2: {:.1f} +/- {:.1f}'.format(np.mean(average_img[:,:,1]), np.std(average_img[:,:,1])))
    print('  chan. 3: {:.1f} +/- {:.1f}'.format(np.mean(average_img[:,:,2]), np.std(average_img[:,:,2])))


if __name__ == "__main__":
    """Choose the appropriate demonstrator data to analyze.

    Note that the demonstration data are generated via:

        python examples/analytic.py oracle --max_episodes=500 --seed=1337

    with appropriate cfg changes + random seed. I then merge lists manually.
    Edit: just use a script for that!
    """
    ## oracle corner policy, pulling at the exact corner edge.
    #fname = 'demos-2019-08-08-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-04-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-03-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    ## oracle corner policy, pulling 'one corner inwards'.
    #fname = 'demos-2019-08-12-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-13-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-12-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    # domain randomization
    #fname = 'demos-2019-08-16-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-16-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl' # note: defective for some reason?
    #fname = 'demos-2019-08-17-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl' # note: use this for tier2
    #fname = 'demos-2019-08-16-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    # heavier domain randomization
    #fname = 'demos-2019-08-20-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-20-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-20-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    # Random baseline
    #fname = 'demos-2019-08-22-pol-random-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-23-pol-random-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-22-pol-random-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    # Highest point baseline
    #fname = 'demos-2019-08-22-pol-highest-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-23-pol-highest-seed-1337_to_1343-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-22-pol-highest-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    # Corner revealer
    ###fname = 'demos-2019-08-27-pol-oracle_reveal-seed-1337_to_1344-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl' # exact corner
    ###fname = 'demos-2019-08-25-pol-oracle_reveal-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl' # exact corner (from adi)
    # one corner inwards for that corner revealer
    #fname = 'demos-2019-08-28-pol-oracle_reveal-seed-1337_to_1344-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-28-pol-oracle_reveal-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-29-pol-oracle_reveal-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_2000_COMBINED.pkl'

    # WRINKLES
    #fname = 'demos-2019-09-01-pol-wrinkle-seed-1000_to_1004-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-09-02-pol-wrinkle-seed-1000_to_1004-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-09-03-pol-wrinkle-seed-1000_to_1004-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_2000_COMBINED.pkl'

    # ------------------------------------------------------------------------------------------------------------------- #
    # USE THESE FOR DEMONSTRATION DATA!!
    # ------------------------------------------------------------------------------------------------------------------- #

    # DEPTH! (And darn I should have added that to the file name ... but I can check with this script, at least!!)
    #fname = 'demos-2019-08-28-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-28-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-28-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'

    # COLOR with correct noise randomization
    #fname = 'demos-2019-08-30-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-30-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_2000_COMBINED.pkl'
    #fname = 'demos-2019-08-30-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_2000_COMBINED.pkl'

    # Now for Feb 2020+, RGBD.
    #fname = 'demos-2020-02-09-16-31-pol-oracle-seed-1336_to_1340-obs-blender-depth-False-rgbd-True-tier1_epis_2000_COMBO.pkl'
    #fname = 'demos-2020-02-10-15-02-pol-oracle-seed-1336_to_1340-obs-blender-depth-False-rgbd-True-tier2_epis_2000_COMBO.pkl'
    #fname = 'demos-2020-02-10-15-05-pol-oracle-seed-1336_to_1340-obs-blender-depth-False-rgbd-True-tier3_epis_2000_COMBO.pkl'

    # ADJUST!!!! For example, if using wrinkles based policy.
    fname = 'demos-2020-02-23-13-53-pol-oracle-seed-1337-obs-blender-depth-False-rgbd-True-tier3_epis_10.pkl'
    USE_IMG = True

    fname = join('logs', fname)
    with open(fname, 'rb') as fh:
        data = pickle.load(fh)
    lengths_obs = np.array([len(x['obs']) for x in data])
    print('the observations are of length:\n{}'.format(lengths_obs))
    lengths = np.array([len(x['rew']) for x in data])
    analyze(data, lengths, use_img=USE_IMG)
