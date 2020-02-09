"""Use this for our analysis scripts (so it uses `demo_baselines.yaml`).

Use `demo_spaces.py` for other debugging.
"""
import subprocess
import pkg_resources
import numpy as np
import argparse
import os
from os.path import join
import sys
import time
import yaml
import logging
import pickle
import datetime
import cv2
from gym_cloth.envs import ClothEnv
from collections import defaultdict
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)

#Adi: Now adding the 'oracle_reveal' demonstrator policy which in reveals occluded corners.
POLICIES = ['oracle','harris','wrinkle','highest','random', 'oracle_reveal']
RAD_TO_DEG = 180. / np.pi
DEG_TO_RAD = np.pi / 180.
BLUE  = (255,0,0)
GREEN = (0,255,0)
RED   = (0,0,255)


class Policy(object):

    def __init__(self):
        pass

    def get_action(self, obs, t):
        raise NotImplementedError()

    def set_env_cfg(self, env, cfg):
        self.env = env
        self.cfg = cfg

    def _data_delta(self, pt, targx, targy, shrink=True):
        """Given pt and target locations, return info needed for action.

        Assumes DELTA actions. Returns x, y of the current point (which should
        be the target) but also the cx, and cy, which should be used if we are
        'clipping' it into [-1,1], but for the 80th time, this really means
        _expanding_ the x,y.
        """
        x, y = pt.x, pt.y
        cx = (x - 0.5) * 2.0
        cy = (y - 0.5) * 2.0
        dx = targx - x
        dy = targy - y
        dist = np.sqrt( (x-targx)**2 + (y-targy)**2 )
        # ----------------------------------------------------------------------
        # Sometimes we grab the top, and can 'over-pull' toward a background
        # corner. Thus we might as well try and reduce it a bit. Experiment!  I
        # did 0.95 for true corners, but if we're pulling one corner 'inwards'
        # then we might want to try a smaller value, like 0.9.
        # ----------------------------------------------------------------------
        if shrink:
            dx *= 0.90
            dy *= 0.90
        return (x, y, cx, cy, dx, dy, dist)


class OracleCornerPolicy(Policy):

    def __init__(self):
        """Oracle corner based policy, cheating as we know the position of points.

        Note the targets, expressed as (x,y):
          upper right: (1,1)
          lower right: (1,0)
          lower left:  (0,0)
          upper left:  (0,1)
        The order in which we pull is important, though!  Choose the method to
        be rotation or distance-based. The latter seems to be more reasonable:
        pick the corner that is furthest from its target.

        Use `np.arctan2(deltay,deltax)` for angle in [-pi,pi] if we use angles.
        Be careful about the action parameterization and if we clip or not.  If
        clipping, we have to convert the x and y to each be in [0,1].

        For tier2 we may have different corner targets for a given point index.
        """
        super().__init__()
        #self._method = 'rotation'
        self._method = 'distance'

    def get_action(self, obs, t):
        """Analytic oracle corner policy.
        """
        if self.cfg['env']['delta_actions']:
            return self._corners_delta(t)
        else:
            return self._corners_nodelta(t)

    def _corners_delta(self, t):
        """Corner-based policy, assuming delta actions.
        """
        pts = self.env.cloth.pts
        assert len(pts) == 625, len(pts)
        cloth = self.env.cloth
        if self.cfg['init']['type'] == 'tier2' and (not cloth.init_side):
            self._ll = 576  # actual corner: 600
            self._ul = 598  # actual corner: 624
            self._lr = 26   # actual corner: 0
            self._ur = 48   # actual corner: 24
            print('NOTE! Flip the corner indices due to init side, tier 2')
            print(self._ll, self._ul, self._lr, self._ur)
        else:
            self._ll = 26   # actual corner: 0
            self._ul = 48   # actual corner: 24
            self._lr = 576  # actual corner: 600
            self._ur = 598  # actual corner: 624
            print('Corners are at the usual indices.')
            print(self._ll, self._ul, self._lr, self._ur)
        x0, y0, cx0, cy0, dx0, dy0, dist0 = self._data_delta(pts[self._ur], targx=1, targy=1)
        x1, y1, cx1, cy1, dx1, dy1, dist1 = self._data_delta(pts[self._lr], targx=1, targy=0)
        x2, y2, cx2, cy2, dx2, dy2, dist2 = self._data_delta(pts[self._ll], targx=0, targy=0)
        x3, y3, cx3, cy3, dx3, dy3, dist3 = self._data_delta(pts[self._ul], targx=0, targy=1)
        maxdist = max([dist0, dist1, dist2, dist3])

        if self._method == 'rotation':
            # Rotate through the corners.
            if t % 4 == 0:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif t % 4 == 1:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif t % 4 == 2:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif t % 4 == 3:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        elif self._method == 'distance':
            # Pick cloth corner furthest from the target.
            if dist0 == maxdist:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif dist1 == maxdist:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif dist2 == maxdist:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif dist3 == maxdist:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        else:
            raise ValueError(self._method)

        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, dx, dy)
        else:
            action = (x, y, dx, dy)
        return action

    def _corners_nodelta(self, t):
        print('Warning! Are you sure you want the no-delta actions?')
        print('We normally do not use this due to pi and -pi angles')

        def _get_data(pt, targx, targy):
            x, y = pt.x, pt.y
            cx = (x - 0.5) * 2.0
            cy = (y - 0.5) * 2.0
            a = np.arctan2(targy-y, targx-x)
            l = np.sqrt( (x-targx)**2 + (y-targy)**2 )
            return (x, y, cx, cy, l, a)

        pts = self.env.cloth.pts
        x0, y0, cx0, cy0, l0, a0 = _get_data(pts[-1], targx=1, targy=1)
        x1, y1, cx1, cy1, l1, a1 = _get_data(pts[-25], targx=1, targy=0)
        x2, y2, cx2, cy2, l2, a2 = _get_data(pts[0], targx=0, targy=0)
        x3, y3, cx3, cy3, l3, a3 = _get_data(pts[24], targx=0, targy=1)
        maxdist = max([l0, l1, l2, l3])

        if self._method == 'rotation':
            # Rotate through the corners.
            if t % 4 == 0:
                x, y, cx, cy, l, a = x0, y0, cx0, cy0, l0, a0
            elif t % 4 == 1:
                x, y, cx, cy, l, a = x1, y1, cx1, cy1, l1, a1
            elif t % 4 == 2:
                x, y, cx, cy, l, a = x2, y2, cx2, cy2, l2, a2
            elif t % 4 == 3:
                x, y, cx, cy, l, a = x3, y3, cx3, cy3, l3, a3
        elif self._method == 'distance':
            # Pick cloth corner furthest from the target.
            if dist0 == maxdist:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif dist1 == maxdist:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif dist2 == maxdist:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif dist3 == maxdist:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        else:
            raise ValueError(self._method)

        # Apply scaling factor to length if needed, since for non-delta actions,
        # length is just the fraction of the maximum number of pulls, which is
        # itself a tuned quantity. Not the same reasoning as the scaling I use
        # for delta actions, but has same effect of reducing pull length.
        l = l * 1.0

        action = (x, y, l, a)
        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, (l-0.5)*2, a/np.pi)
        else:
            action = (x, y, l, a)
        return action

#Adi: Adding the OracleCornerRevealPolicy
#Algorithm:
#1. Find all visible corners a minimum distance away from their target position and pull the visible corner furthest from its respective bed corner
#2. If no corners are visible, perform a revealing action on the non-visible corner furthest from its respective bed corner
#3. Back to step 1
class OracleCornerRevealPolicy(Policy):

    def __init__(self):
        """Oracle corner reveal based policy, cheating as we know the position of points.

        Note the targets, expressed as (x,y):
          upper right: (1,1)
          lower right: (1,0)
          lower left:  (0,0)
          upper left:  (0,1)
        The order in which we pull is important, though!  Choose the method to
        be rotation or distance-based. The latter seems to be more reasonable:
        pick the corner that is furthest from its target.

        Use `np.arctan2(deltay,deltax)` for angle in [-pi,pi] if we use angles.
        Be careful about the action parameterization and if we clip or not.  If
        clipping, we have to convert the x and y to each be in [0,1].

        For tier2 we may have different corner targets for a given point index.
        """
        #super().__init__()
        #self._method = 'rotation'
        self._method = 'distance'
        self._alg = 'alg1'
        self._sign = 1 #This determines if we are pulling or revealing
        self._hd = 200
        self._wd = 200

    def get_action(self, obs, t):
        """Analytic oracle corner policy.
        """
        #Adi: Oracle Corner Reveal Policy only supports delta actions!
        if self.cfg['env']['delta_actions']:
            return self._corners_delta(t)
        else:
            return self._corners_nodelta(t)

    def _corners_delta(self, t):
        """Corner-based policy, assuming delta actions.
        """
        pts = self.env.cloth.pts
        assert len(pts) == 625, len(pts)
        cloth = self.env.cloth
        if self.cfg['init']['type'] == 'tier2' and (not cloth.init_side):
            #self._ll = 600
            #self._ul = 624
            #self._lr = 0
            #self._ur = 24
            self._ll = 576  # actual corner: 600
            self._ul = 598  # actual corner: 624
            self._lr = 26   # actual corner: 0
            self._ur = 48   # actual corner: 24
            print('NOTE! Flip the corner indices due to init side, tier 2')
            print(self._ll, self._ul, self._lr, self._ur)
        else:
            #self._ll = 0
            #self._ul = 24
            #self._lr = 600
            #self._ur = 624
            self._ll = 26   # actual corner: 0
            self._ul = 48   # actual corner: 24
            self._lr = 576  # actual corner: 600
            self._ur = 598  # actual corner: 624
            print('Corners are at the usual indices.')
            print(self._ll, self._ul, self._lr, self._ur)
        x0, y0, cx0, cy0, dx0, dy0, dist0 = self._data_delta(pts[self._ur], targx=1, targy=1)
        x1, y1, cx1, cy1, dx1, dy1, dist1 = self._data_delta(pts[self._lr], targx=1, targy=0)
        x2, y2, cx2, cy2, dx2, dy2, dist2 = self._data_delta(pts[self._ll], targx=0, targy=0)
        x3, y3, cx3, cy3, dx3, dy3, dist3 = self._data_delta(pts[self._ul], targx=0, targy=1)
        maxdist = max([dist0, dist1, dist2, dist3])

        #Adi: Print occlusion vector for debugging purposes
        print("Occlusion Vector:")
        print(self.env._occlusion_vec)

        distances = [dist0, dist1, dist2, dist3]
        visible_corner_indices = [i for i, v in enumerate(self.env._occlusion_vec) if not v]
        occluded_corner_indices = [i for i, v in enumerate(self.env._occlusion_vec) if v]

        distances_visible = [d if i in visible_corner_indices else 0 for i, d in enumerate(distances)]
        distances_occluded = [d if i in occluded_corner_indices else 0 for i, d in enumerate(distances)]

        maxdist_visible = max(distances_visible)
        maxdist_occluded = max(distances_occluded)

        thresh_dist = 0.09

        if self._alg == 'alg1':
            #Perform the algorithm described above
            #Check if there are any visible corners that are beyond a certain threshold distance from their respective bed corner:
            if (False in self.env._occlusion_vec and maxdist_visible > thresh_dist) or (True not in self.env._occlusion_vec and maxdist_visible < thresh_dist):
                #Perform Step 1:
                #Find the visible corner furthest from its respective bed corner:
                print("Visible Corner Indices:")
                print(visible_corner_indices)
                maxdist = maxdist_visible
                dist0 = distances_visible[0]
                dist1 = distances_visible[1]
                dist2 = distances_visible[2]
                dist3 = distances_visible[3]
                self._sign = 1
            else:
                #Perform Step 2:
                #Since all corners are occluded, perform a revealing action on the furthest occluded corner
                print("Occluded Corner Indices:")
                print(occluded_corner_indices)
                maxdist = maxdist_occluded
                dist0 = distances_occluded[0]
                dist1 = distances_occluded[1]
                dist2 = distances_occluded[2]
                dist3 = distances_occluded[3]
                self._sign = -1

        distances = [dist0, dist1, dist2, dist3]
        print(distances)


        if self._method == 'distance':
            # Pick cloth corner furthest from the target.
            if dist0 == maxdist:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif dist1 == maxdist:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif dist2 == maxdist:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif dist3 == maxdist:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        else:
            raise ValueError(self._method)

        #If we are performing a revealing action, let's fix the distance we reveal for now
        if self._sign == -1:
            scaling_factor = 0.5 / maxdist
            dx *= scaling_factor
            dy *= scaling_factor


        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, dx * self._sign, dy * self._sign)
        else:
            action = (x, y, dx * self._sign, dy * self._sign)
        return action


class HarrisCornerPolicy(Policy):
    """Note: strongly recommended we don't do this. It will fail miserably.

    Just show in an appendix that the HCD detects way too many corners, and
    cite our prior paper as well for why it's a bad idea.
    """

    def __init__(self):
        """Harris Corner Detector policy, so we are not 'cheating'.

        I suggest the same hyperparameters as we did in the bed-making paper.
        The next (tricky) part is figuring out which _action_ to take, given
        the image, because we have to be given a set of candidate corners.
        Intuitively, we should figure out ways to determine if corners are the
        'most bottom-left', the 'most bottom-right', and so on, and then we can
        assign a target using the oracle corner-based policy. And THEN we have
        the problem of converting pixel-space points to one on the cloth.

        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
        Default args: blockSize=2, ksize=3, k=0.04.  There's also a 0.01 value
        which is `self.thresh` here.

        We reduced ksize=1 in the bed-making paper for 'higher sensitivity' but
        I am not actually sure if that helped.  Note, ksize is the 'kernel'
        size (integer), and it must be odd and less than 31. blockSize also
        represents a size/area-like quantity, but can be even. I'm not sure
        which of blockSize or ksize is used in the documentation for
        determining E[u,v].

        Changing ksize doesn't seem to impact number of corners found.
        Increasing blockSize appears to increase number of corners, but mostly
        crowded around similar regions, so no 'new' areas.

        I suppose we could also change the 'dst > 0.01 * dst.max()' by
        increasing the 0.01? That means fewer corners pass the threshold.
        """
        super().__init__()
        self.blockSize = 2
        self.ksize = 1
        self.k = 0.04
        self.thresh = 0.10

    def _get_obs_fname(self):
        """Convenience method for file names if we need to save observations.
        """
        directory = '.'
        k = len([x for x in os.listdir(directory) if '.png' in x])
        fname = 'obs_{}.png'.format(str(k).zfill(3))
        return fname

    def _harris(self, img_h):
        """Actually apply the Harris corner detection.

        Debug + check images. We might need a filter to get rid of corners due
        to the background plane (if it exists) that we want to cover.  Also, we
        might want to 'crop' the valid range for the corners. We used a naive
        'grid' method where we built a rectangle and use only corners in that.

        Returns: (corners, closest_corners, image)
        """
        add_all_corners_to_img = False
        all_blue_corners = True

        # Harris Corner Detector, from online tutorials.
        gray = cv2.cvtColor(img_h, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst  = cv2.cornerHarris(gray, blockSize=self.blockSize, ksize=self.ksize, k=self.k)
        dst  = cv2.dilate(dst, None)
        if add_all_corners_to_img:
            img_h[dst > self.thresh * dst.max()] = [0,0,255]

        # Collect corners so we can add them to image in our preferred manner.
        corners = np.where(dst > self.thresh * dst.max())
        cx, cy = corners
        assert cx.shape == cy.shape, "{} vs {}".format(cx.shape, cy.shape)
        print('number of corners:', cx.shape, cy.shape)
        corners = np.concatenate((cx[None],cy[None]))
        assert len(corners.shape) == 2 and corners.shape[0] == 2  # shape = (2, num_corners)

        # Filter corners and get _indices_ of remaining ones.
        xlarge = np.where( 0 < corners[1,:] )[0]
        ylarge = np.where( 0 < corners[0,:] )[0]
        xsmall = np.where( corners[1,:] < 10000 )[0]
        ysmall = np.where( corners[0,:] < 10000 )[0]
        xfilter = np.intersect1d(xsmall, xlarge)
        yfilter = np.intersect1d(ysmall, ylarge)
        filt_corners_indices = np.intersect1d(xfilter, yfilter)

        # --------------------------------------------------------------------------
        # Draw the filtered corners. Note:
        #   img.shape == (h, w, 3), assuming rgb
        #   corners.shape == (2, num_corners)
        #   corners[0,:] ranges from (0 to h)
        #   corners[1,:] ranges from (0 to w)
        # Thus, corners[k] is the k-th indexed corner (length 2 tuple), with
        # the respective corner indices within (0,h) and (0,w).  But when you
        # view the image like a human views it, the first corner coordinate is
        # the y axis, but going DOWNWARDS. The second corner coordinate is the
        # usual x axis, going rightwards.
        # --------------------------------------------------------------------------

        # Also return the bottom left/right and upper left/right, wrt pixels.
        # We want to paste these on images, so want (corners[1], corners[0]).
        # These are passing my sanity checks, so at least that's good.
        closest = {
            'br': (None, np.inf),
            'bl': (None, np.inf),
            'ur': (None, np.inf),
            'ul': (None, np.inf),
        }
        H, W, C = img_h.shape

        for idx in filt_corners_indices:
            corner = corners[:,idx]
            # Only do this if I want to see all corners.
            if add_all_corners_to_img:
                img_h[ corner[0], corner[1] ] = [255, 0, 0]

            # We have to flip the coords for writing on images with cv2.
            if all_blue_corners:
                c0 = corner[1]
                c1 = corner[0]
                w = 4
                cv2.line(img_h, (c0-w, c1-w), (c0+w, c1+w), color=BLUE, thickness=1)
                cv2.line(img_h, (c0+w, c1-w), (c0-w, c1+w), color=BLUE, thickness=1)

            # Bottom right corner
            dist = (H - corner[0])**2 + (W - corner[1])**2
            if dist < closest['br'][1]:
                closest['br'] = ( (corner[1],corner[0]), dist )
            # Bottom left corner
            dist = (H - corner[0])**2 + (0 - corner[1])**2
            if dist < closest['bl'][1]:
                closest['bl'] = ( (corner[1],corner[0]), dist )
            # Upper right corner
            dist = (0 - corner[0])**2 + (W - corner[1])**2
            if dist < closest['ur'][1]:
                closest['ur'] = ( (corner[1],corner[0]), dist )
            # Upper left corner
            dist = (0 - corner[0])**2 + (0 - corner[1])**2
            if dist < closest['ul'][1]:
                closest['ul'] = ( (corner[1],corner[0]), dist )

        # Label more corners on image if desired for debugging.
        cnr = closest['bl'][0]  # NOTE/TODO pick one of the four
        w = 8
        cv2.line(img_h, (cnr[0]-w, cnr[1]-w), (cnr[0]+w, cnr[1]+w),
                 color=GREEN, thickness=1)
        cv2.line(img_h, (cnr[0]+w, cnr[1]-w), (cnr[0]-w, cnr[1]+w),
                 color=GREEN, thickness=1)
        #cv2.circle(img_h, center=___, radius=22, color=BLUE, thickness=4)
        #cv2.circle(img_h, center=___, radius=22, color=RED, thickness=4)

        return corners, closest, img_h

    def get_action(self, obs, t):
        """Given a corner detector figure out what action to take.
        """
        obs_h = obs.copy()
        corners_h, closest_corners_h, obs_h, = self._harris(obs_h)
        _, num_corners = corners_h.shape  # shape = (2, num_corners)

        # Save observation, original and one with the predicted corners on it.
        cv2.imwrite(self._get_obs_fname(), obs)
        fname = self._get_obs_fname()
        _debuginfo = '_corners_{}_block_{}_ksize_{}_k_{}.png'.format(num_corners,
                self.blockSize, self.ksize, self.k)
        fname = fname.replace('.png',_debuginfo)
        cv2.imwrite(fname, obs_h)

        print("Warning! We do not have the action fully implemented for the HCD " +
            "We are doing actions at random, because we need a mapping from a " +
            "predicted corner in image space to the cloth (x,y) coordinate.")
        return self.env.get_random_action(atype='over_xy_plane')

        # End of debugging, compute the action. We get targx, targy by picking
        # the corner that is assumed to be furthest away from the target.
        # HOWEVER, we also have to ensure that we get a correct mapping as
        # stated earlier, which means finding the correct `pt` here. Use
        # closest_corners_h to get the four estimated corners.
        targx = 0  # fix
        targy = 0  # fix
        x, y, cx, cy, dx, dy, dist = self._data_delta(pt, targx, targy)
        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, dx, dy)
        else:
            action = (x, y, dx, dy)
        return action


class WrinklesPolicy(Policy):

    def __init__(self):
        super().__init__()

    def get_action(self, obs, t):
        raise NotImplementedError()


class HighestPointPolicy(Policy):

    def __init__(self):
        """Highest point, with known targets.

        It is proabably best for this method if we can map from a given point
        to where its location should be on a fully flat cloth. We shouldn't be
        pulling highest points 'blindly' to a given corner because often the
        highest points correspond to central locations on the cloth. Of course,
        this means we are cheating again, like with the oracle-based policy,
        but hey, we have to strengthen the baseline methods as much as we can.

        Also, I suggest we do a stochastic version where we randomly choose
        among any of the highest points. That makes it more resistant to
        getting trapped in local minima.
        """
        super().__init__()
        self.top_k = 5

    def _get_targ_xy(self, pt_idx, pt):
        """Determine (x,y) of a given index, assuming flat cloth grid.

        For tiers 1 and 3 it's easy, we saved the original point (x,y)
        positions, and that was on a flat grid. For tier 2:

        - If init_side=True then we dropped from left side, so all we do is
          convert the z value to x, to get the position. The y is the same.
        - Else, we dropped from the right side. The y-value is the same as
          usual but for x, convert z value so it's 1-z.

        I used to do this:

        - We have a 25x25 grid of cloth, and the spacing in dx and y is 1/24,
          see https://github.com/BerkeleyAutomation/gym-cloth/issues/29.
        - For x: indices 0-24 are first 'column', then 25-49 are second, and so
          on, up to 600-624 for last column. When we divide by 25 and take
          ints, we get integers from 0 to 24. But then we actually need to
          divide by 24 to map to the interval [0,1] correctly. Indices 0-24 map
          to x values of 0, and 600-624 maps to x values of 1.
        - For y: instead of dividing by 25, we mod it, to get {0,25,50,...,600}
          to the same value, and similarly for others. Then, like in the x
          case, divide by 24 to get the full range [0,1].

        but it's easier to just get a simple target.
        """
        if self.cfg['init']['type'] in ['tier1', 'tier3']:
            x = pt.orig_x
            y = pt.orig_y
            z = pt.orig_z
            print('tier {}, originally ({:.3f},{:.3f},{:.3f})'.format(
                    self.cfg['init']['type'], x, y, z))
        else:
            #x = int(pt_idx / 25) / 24.
            #y = (pt % 25) / 24.
            x = pt.orig_x
            y = pt.orig_y
            z = pt.orig_z
            print('tier {}, originally at ({:.3f},{:.3f},{:.3f})'.format(
                    self.cfg['init']['type'], x, y, z))
            if self.env.cloth.init_side:
                x = pt.orig_z
                y = pt.orig_y
                print('init_side=True, thus our target: ({:.3f},{:.3f})'.format(x,y))
            else:
                x = 1.0 - pt.orig_z
                y = pt.orig_y
                print('init_side=False, thus our target: ({:.3f},{:.3f})'.format(x,y))
        return (x,y)

    def get_action(self, obs, t):
        """Get highest point, and figure out direction.
        """
        assert self.cfg['env']['delta_actions']
        sorted_points = sorted(
                [p for p in self.env.cloth.pts], key=lambda p:p.z, reverse=True)
        pt = sorted_points[ np.random.randint(self.top_k) ]
        pt_idx = (self.env.cloth.pts).index(pt)
        print('Top k sorted: {}'.format(sorted_points[:self.top_k]))
        print('Selecting point {}, at overall index {}'.format(pt, pt_idx))
        targx, targy = self._get_targ_xy(pt_idx, pt)
        x, y, cx, cy, dx, dy, dist = self._data_delta(pt, targx, targy)
        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, dx, dy)
        else:
            action = (x, y, dx, dy)
        return action


class RandomPolicy(Policy):

    def __init__(self):
        """Two possible types of random policies, pick one.

        Should work for all the cloth tiers.
        """
        super().__init__()
        self.type = 'over_xy_plane'

    def get_action(self, obs, t):
        return self.env.get_random_action(atype=self.type)


def run(args, policy):
    """Run an analytic policy, using similar setups as baselines-fork.

    If we have a random seed in the args, we use that instead of the config
    file. That way we can run several instances of the policy in parallel for
    faster data collection.
    """
    with open(args.cfg_file, 'r') as fh:
        cfg = yaml.safe_load(fh)
        if args.seed is not None:
            seed = args.seed
            cfg['seed'] = seed  # Actually I don't think it's needed but doesn't hurt?
        else:
            seed = cfg['seed']
        if seed == 1500 or seed == 1600:
            print('Ideally, avoid using these two seeds.')
            sys.exit()
        assert cfg['env']['clip_act_space'] and cfg['env']['delta_actions']
        stuff = '-seed-{}-obs-{}-depth-{}-rgbd-{}-{}_epis_{}'.format(seed,
                    cfg['env']['obs_type'],
                    cfg['env']['use_depth'],
                    cfg['env']['use_rgbd'],
                    cfg['init']['type'],
                    args.max_episodes
        )
        result_path = args.result_path.replace('.pkl', '{}.pkl'.format(stuff))
        assert not cfg['env']['force_grab'], 'Do not need force_grab for analytic'
        print('\nOur result_path:\n\t{}'.format(result_path))
    np.random.seed(seed)

    # Should seed env this way, following gym conventions.  NOTE: we pass in
    # args.cfg_file here, but then it's immediately loaded by ClothEnv. When
    # env.reset() is called, it uses the ALREADY loaded parameters, and does
    # NOT re-query the file again for parameters (that'd be bad!).
    env = ClothEnv(args.cfg_file)
    env.seed(seed)
    env.render(filepath=args.render_path)
    policy.set_env_cfg(env, cfg)

    # Book-keeping.
    num_episodes = 0
    stats_all = []
    coverage = []
    variance_inv = []
    nb_steps = []

    for ep in range(args.max_episodes):
        obs = env.reset()
        # Go through one episode and put information in `stats_ep`.
        # Don't forget the first obs, since we need t _and_ t+1.
        stats_ep = defaultdict(list)
        stats_ep['obs'].append(obs)
        done = False
        num_steps = 0

        while not done:
            action = policy.get_action(obs, t=num_steps)
            obs, rew, done, info = env.step(action)
            stats_ep['obs'].append(obs)
            stats_ep['rew'].append(rew)
            stats_ep['act'].append(action)
            stats_ep['done'].append(done)
            stats_ep['info'].append(info)
            num_steps += 1
        num_episodes += 1
        coverage.append(info['actual_coverage'])
        variance_inv.append(info['variance_inv'])
        nb_steps.append(num_steps)
        stats_all.append(stats_ep)
        print("\nInfo for most recent episode: {}".format(info))
        print("Finished {} episodes.".format(num_episodes))
        print('  {:.2f} +/- {:.1f} (coverage)'.format(
                np.mean(coverage), np.std(coverage)))
        print('  {:.2f} +/- {:.1f} ((inv)variance)'.format(
                np.mean(variance_inv), np.std(variance_inv)))
        print('  {:.2f} +/- {:.1f} (steps per episode)'.format(
                np.mean(nb_steps), np.std(nb_steps)))

        # Just dump here to keep saving and overwriting.
        with open(result_path, 'wb') as fh:
            pickle.dump(stats_all, fh)

    assert len(stats_all) == args.max_episodes, len(stats_all)
    if env.render_proc is not None:
        env.render_proc.terminate()
        env.cloth.stop_render()


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument("policy", type=str, help="name of the policy to use")
    pp.add_argument("--max_episodes", type=int, default=10)
    pp.add_argument("--seed", type=int)
    pp.add_argument("--tier", type=int)
    args = pp.parse_args()
    assert args.tier in [1,2,3], args.tier

    args.policy = (args.policy).lower()
    if args.policy == 'oracle':
        policy = OracleCornerPolicy()
    elif args.policy == 'harris':
        policy = HarrisCornerPolicy()
    elif args.policy == 'wrinkle':
        policy = WrinklesPolicy()
    elif args.policy == 'highest':
        policy = HighestPointPolicy()
    elif args.policy == 'random':
        policy = RandomPolicy()
    elif args.policy == 'oracle_reveal':
        policy = OracleCornerRevealPolicy()
    else:
        raise ValueError(args.policy)

    # Use this to store results. For example, these can be used to save the
    # demonstrations that we later load to augment DeepRL training. We can
    # augment the file name later in `run()`. Add policy name so we know the
    # source. Fortunately, different trials can be combined in a larger lists.
    date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    result_pkl = 'demos-{}-pol-{}.pkl'.format(date, args.policy)

    # Each time we use the environment, we need to pass in some configuration.
    args.file_path = fp = os.path.dirname(os.path.realpath(__file__))
    #args.cfg_file = join(fp, '../cfg/demo_baselines.yaml') # BASELINES!
    args.cfg_file = join(fp, '../cfg/t{}_rgbd.yaml'.format(args.tier))
    args.render_path = join(fp, '../render/build')    # Must be compiled!
    args.result_path = join(fp, '../logs/{}'.format(result_pkl))

    run(args, policy)
