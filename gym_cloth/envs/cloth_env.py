"""
An OpenAI Gym-style environment for the cloth smoothing experiments. It's not
exactly their interface because we pass in a configuration file. See README.md
document in this directory for details.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
from os.path import join
import subprocess
import sys
import time
import datetime
import logging
import json
import yaml
import subprocess
import trimesh
import cv2
import datetime
import pickle
import copy
import pkg_resources
from gym_cloth.physics.cloth import Cloth
from gym_cloth.physics.point import Point
from gym_cloth.physics.gripper import Gripper
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.spatial import ConvexHull

_logging_setup_table = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
}

# Thresholds for successful episode completion.
# Keys are possible values for reward_type in config.
_REWARD_THRESHOLDS = {
    'coverage': 0.92,
    'coverage-delta': 0.92,
    'height': 0.85,
    'height-delta': 0.85,
    'variance': 2,
    'variance-delta': 2,
    'folding-number': 0
}

_EPS = 1e-5


class ClothEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, subrank=None, start_state_path=None):
        """Various initialization for the environment.

        Not to be confused with the initialization for the _cloth_.

        See the following for how to use proper seeding in gym:
          https://github.com/openai/gym/blob/master/gym/utils/seeding.py
          https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py
          https://stackoverflow.com/questions/5836335/
          https://stackoverflow.com/questions/22994423/

        The bounds of (1,1,1) are for convenience and should be the same bounds
        as the cloth has internally.

        RL algorithms refer to observation_space and action_space when building
        neural networks.  Also, we may need to sample from our action space for
        a random policy.  For the actions, we should enforce clipping, but it's
        a subtle issue. See how others do it for MuJoCo.

        Optional:
        - `subrank` if we have multiple envs in parallel, to make it easy to
          tell which env corresponds to certain loggers.
        - `start_state_path` if we want to force cloth to start at a specific
          state.  Represents the path to the state file.
        """
        # NOTE! This is the only time we load from the yaml file. We do this
        # when ClothEnv is called, and it uses the fixed values from the yaml
        # file. Changing the yaml file later while code is running is OK as it
        # will not affect results.
        with open(cfg_file, 'r') as fh:
            cfg = yaml.safe_load(fh)
        self.cfg              = cfg
        self.cfg_file         = cfg_file
        self.max_actions      = cfg['env']['max_actions']
        self.max_z_threshold  = cfg['env']['max_z_threshold']
        self.iters_up         = cfg['env']['iters_up']
        self.iters_up_rest    = cfg['env']['iters_up_rest']
        self.iters_pull_max   = cfg['env']['iters_pull_max']
        self.iters_grip_rest  = cfg['env']['iters_grip_rest']
        self.iters_rest       = cfg['env']['iters_rest']
        self.updates_per_move = cfg['env']['updates_per_move']
        self.reduce_factor    = cfg['env']['reduce_factor']
        self.grip_radius      = cfg['env']['grip_radius']
        self.render_gl        = cfg['init']['render_opengl']
        self._init_type       = cfg['init']['type']
        self._clip_act_space  = cfg['env']['clip_act_space']
        self._delta_actions   = cfg['env']['delta_actions']
        self._bilateral       = cfg['env']['bilateral_actions']
        self._obs_type        = cfg['env']['obs_type']
        self._force_grab      = cfg['env']['force_grab']
        self._oracle_reveal   = cfg['env']['oracle_reveal']
        self._use_depth       = cfg['env']['use_depth']
        self._use_rgbd        = cfg['env']['use_rgbd']
        self._radius_inc      = 0.02
        self.bounds = bounds  = (1, 1, 1)
        self.render_proc      = None
        self.render_port      = 5556
        self._logger_idx      = subrank
        self._occlusion_vec   = [True, True, True, True]
        self.__add_dom_rand   = cfg['env']['use_dom_rand']              # string
        self._add_dom_rand    = (self.__add_dom_rand.lower() == 'true') # boolean
        self.dom_rand_params  = {} # Ryan: store domain randomization params to keep constant per episode

        if start_state_path is not None:
            with open(start_state_path, 'rb') as fh:
                self._start_state = pickle.load(fh)
        else:
            self._start_state = None

        # Reward design. Very tricky. BE EXTRA CAREFUL WHEN ADJUSTING,
        # particularly if we do DeepRL with demos, because demos need rewards
        # consistent with those seen during training.
        self.reward_type = cfg['env']['reward_type']
        assert 'coverage' in self.reward_type
        self._prev_reward = 0
        self._neg_living_rew = 0.0     # -0.05
        self._nogrip_penalty = -0.01   # I really think we should have this
        self._tear_penalty = 0.0       # -10 (OpenAI Dactyl -10 for failures)
        self._oob_penalty = 0.0        # -10 (OpenAI Dactyl -10 for failures)
        self._cover_success = 5.       # (OpenAI Dactyl +5 for success)
        self._act_bound_factor = 1.0
        self._act_pen_limit = 3.0
        self._current_coverage = 0.0

        # Create observation ('1d', '3d') and action spaces. Possibly make the
        # obs_type and other stuff user-specified parameters.
        self._slack = 0.25
        self.num_w = num_w = cfg['cloth']['num_width_points']
        self.num_h = num_h = cfg['cloth']['num_height_points']
        self.num_points = num_w * num_h
        lim = 100
        if self._obs_type == '1d':
            self.obslow  = np.ones((3 * self.num_points,)) * -lim
            self.obshigh = np.ones((3 * self.num_points,)) * lim
            self.observation_space = spaces.Box(self.obslow, self.obshigh)
        elif self._obs_type == 'blender':
            self._hd = 224
            self._wd = 224
            self.obslow = np.zeros((self._hd, self._wd, 3)).astype(np.uint8)
            self.obshigh = np.ones((self._hd, self._wd, 3)).astype(np.uint8)
            self.observation_space = spaces.Box(self.obslow, self.obshigh, dtype=np.uint8)
        else:
            raise ValueError(self._obs_type)

        # Ideally want the gripper to grip points in (0,1) for x and y. Perhaps
        # consider slack that we use for out of bounds detection? Subtle issue.
        b0, b1 = self.bounds[0], self.bounds[1]
        if self._bilateral:
            assert self._clip_act_space # only compatible with clip_act_space for now
        if self._clip_act_space:
            # Applies regardless of 'delta actions' vs non deltas.  Note misleading
            # 'clip' name, because (0,1) x/y-coords are *expanded* to (-1,1).
            self.action_space = spaces.Box(
                low= np.array([-1., -1., -1., -1.]),
                high=np.array([ 1.,  1.,  1.,  1.])
            )
        else:
            if self._delta_actions:
                self.action_space = spaces.Box(
                    low= np.array([0., 0., -1., -1.]),
                    high=np.array([1., 1.,  1.,  1.])
                )
            else:
                self.action_space = spaces.Box(
                    low= np.array([  -self._slack,   -self._slack, 0.0, -np.pi]),
                    high=np.array([b0+self._slack, b1+self._slack, 1.0,  np.pi])
                )

        # Bells and whistles
        self._setup_logger()
        self.seed()
        self.debug_viz = cfg['init']['debug_matplotlib']

    @property
    def state(self):
        """Returns state representation as a numpy array.

        The heavy duty part of this happens if we're using image observations,
        in which we form the mesh needed for Blender to render it. We also pass
        relevant arguments to the Blender script.
        """
        if self._obs_type == '1d':
            lst = []
            for pt in self.cloth.pts:
                lst.extend([pt.x, pt.y, pt.z])
            return np.array(lst)
        elif self._obs_type == 'blender':
            # Ryan: implement RGBD
            if self._use_rgbd == 'True':
                img_rgb = self.get_blender_rep('False')
                img_d = self.get_blender_rep('True')[:,:,0]
                return np.dstack((img_rgb, img_d))
            else:
                return self.get_blender_rep(self._use_depth)
        else:
            raise ValueError(self._obs_type)

    def get_blender_rep(self, use_depth):
        """Ryan: put get_blender_rep in its own method so we can easily get RGBD images."""
        bhead = '/tmp/blender'
        if not os.path.exists(bhead):
            os.makedirs(bhead)

        # Step 1: make obj file using trimesh, and save to directory.
        wh = self.num_w
        #wh = self.num_h
        assert self.num_w == self.num_h  # TODO for now
        cloth = np.array([[p.x, p.y, p.z] for p in self.cloth.pts])
        assert cloth.shape[1] == 3, cloth.shape
        faces = []
        for r in range(wh-1):
            for c in range(wh-1):
                pp = r*wh + c
                faces.append( [pp,   pp+wh, pp+1] )
                faces.append( [pp+1, pp+wh, pp+wh+1] )
        tm = trimesh.Trimesh(vertices=cloth, faces=faces)
        # Handle file naming. Hopefully won't be duplicates!
        rank = '{}'.format(self._logger_idx)
        step = '{}'.format(str(self.num_steps).zfill(3))
        date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        base = 'gym-cloth-r{}-s{}-{}'.format(rank, step, date)
        tm_path = join(bhead, base)
        randnum = np.random.randint(1000000)  # np.random instead of np_random :-)
        tm_path = '{}_r{}.obj'.format(tm_path, str(randnum).zfill(7))
        tm.export(tm_path)

        # Step 2: call blender to get image representation.  We assume the
        # `blender` sub-package is at the same level as `envs`.  Use
        # __dirname__ to get path, then switch to `blender` dir.  Also deal
        # with data paths (for background frame) and different machines.

        init_side = 1 if self.cloth.init_side else -1
        #bfile = join(os.path.dirname(__file__), '../blender/get_image_rep.py') # 2.80
        bfile = join(os.path.dirname(__file__), '../blender/get_image_rep_279.py')
        frame_path = pkg_resources.resource_filename('gym_cloth', 'blender/frame0.obj')
        floor_path = pkg_resources.resource_filename('gym_cloth', 'blender/floor.obj')

        #Adi: Adding argument/flag for the oracle_reveal demonstrator
        #Adi: Adding argument/flag for using depth images
        #Adi: Adding argument for the floor obj path for more accurate depth images
        #Adi: Adding flag for domain randomization
        #Ryan: Adding hacky flags for fixed dom rand params per episode
        if sys.platform == 'darwin':
            subprocess.call([
                '/Applications/Blender/blender.app/Contents/MacOS/blender',
                '--background', '--python', bfile, '--', tm_path,
                str(self._hd), str(self._wd), str(init_side), self._init_type,
                frame_path, self._oracle_reveal, use_depth, floor_path,
                self.__add_dom_rand,
                ",".join([str(i) for i in self.dom_rand_params['c']]),
                ",".join([str(i) for i in self.dom_rand_params['n1']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_pos']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_deg']]),
                str(self.dom_rand_params['specular_max'])
                ]
            )
        else:
            subprocess.call([
                'blender', '--background', '--python', bfile, '--', tm_path,
                str(self._hd), str(self._wd), str(init_side), self._init_type,
                frame_path, self._oracle_reveal, use_depth, floor_path,
                self.__add_dom_rand,
                ",".join([str(i) for i in self.dom_rand_params['c']]),
                ",".join([str(i) for i in self.dom_rand_params['n1']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_pos']]),
                ",".join([str(i) for i in self.dom_rand_params['camera_deg']]),
                str(self.dom_rand_params['specular_max'])
                ]
            )
        time.sleep(1)  # Wait a bit just in case

        # Step 3: load image from directory saved by blender.
        #Adi: Loading the occlusion state as well and saving it
        blender_path = tm_path.replace('.obj','.png')
        occlusion_path_pkl = tm_path.replace('.obj', '')
        with open(occlusion_path_pkl, 'rb') as fp:
            itemlist = pickle.load(fp)
            self._occlusion_vec = itemlist

        img = cv2.imread(blender_path)
        assert img.shape == (self._hd, self._wd, 3), \
                'error, shape {}, idx {}'.format(img.shape, self._logger_idx)

        if use_depth == 'True':
            # Smooth the edges b/c of some triangles.
            img = cv2.bilateralFilter(img, 7, 50, 50)
            if self._add_dom_rand:
                gval = self.dom_rand_params['gval_depth']
                img = np.uint8( np.maximum(0, np.double(img)-gval) )
            else:
                img = np.uint8( np.maximum(0, np.double(img)-50) )
        else:
            # Might as well randomize brightness if we're doing RGB.
            if self._add_dom_rand:
                gval = self.dom_rand_params['gval_rgb']
                img = self._adjust_gamma(img, gamma=gval)

        if self._add_dom_rand:
            # Apply some noise ONLY AT THE END OF EVERYTHING. I think it's
            noise = self.dom_rand_params['noise']
            img = np.minimum( np.maximum(np.double(img)+noise, 0), 255 )
            img = np.uint8(img)

        # If desired, save the images. Also save the resized version  --
        # make sure it's not done before we do noise addition, etc!
        #cv2.imwrite(blender_path, img)
        #img_small = cv2.resize(img, dsize=(100,100))  # dsize=(width,height)
        #cv2.imwrite(tm_path.replace('.obj','_small.png'), img_small)
        # careful! only if we want to override paths!!
        #cv2.imwrite(blender_path, img_small)
        #time.sleep(2)

        # Step 4: remaining book-keeping.
        if os.path.isfile(tm_path):
            os.remove(tm_path)
        return img

    def seed(self, seed=None):
        """Apply the env seed.

        See, for example:
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        We follow a similar convention by using an `np_random` object.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.logger.debug("Just re-seeded env to: {}".format(seed))
        return [seed]

    def save_state(self, cloth_file):
        """Save cloth.pts as a .pkl file.

        Be sure to supply a full path. Otherwise it saves under the `build/`
        directory somewhere.
        """
        with open(cloth_file, 'wb') as fh:
            pickle.dump({"pts": self.cloth.pts, "springs": self.cloth.springs}, fh)

    def _pull(self, i, iters_pull, x_diag_r, y_diag_r):
        """Actually perform pulling, assuming length/angle actions.

        There are two cases when the pull should be stable: after pulling up,
        and after pulling in the plane with a fixed z height.
        """
        if i < self.iters_up:
            self.gripper.adjust(x=0.0, y=0.0, z=0.0025)
        elif i < self.iters_up + self.iters_up_rest:
            pass
        elif i < self.iters_up + self.iters_up_rest + iters_pull:
            self.gripper.adjust(x=x_diag_r, y=y_diag_r, z=0.0)
        elif i < self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest:
            pass
        elif i < self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest + self.iters_rest - 5:
            self.gripper.release()
        else: # release bilateral, if applicable
            self.gripper.release(bilateral=True)

    def step(self, action, initialize=False):
        """Execute one action.

        Currently, actions are parameterized as (grasp_point, pull fraction
        length, pull direction).  It will grasp at some target point, and then
        pull in the chosen direction for some number of cloth updates. We have
        rest periods to help with stability.

        If we clipped the action space into [-1,1], (meaning the policy or
        human would output values between [-1,1] for each component) then for
        actions with true ranges of [0,1], divide the original [-1,1] actions
        by two and add 0.5. For angle, multiply by pi.

        Parameters
        ----------
        action: tuple
            Action to be applied this step.
        initialize: bool
            Normally false. If true, that means we're in the initialization step
            from an `env.reset()` call, and so we probably don't want to count
            these 'actions' as part of various statistics we compute.

        Returns
        -------
        Usual (state, reward, done, info) from env steps. Our info contains the
        number of steps called, both for actions and `cloth.update()` calls.
        """
        info = {}
        logger = self.logger
        exit_early = False
        astr = self._act2str(action)

        # Truncate actions according to our bounds, then grip.
        low  = self.action_space.low
        high = self.action_space.high
        if self._bilateral:
            # bilateral actions are (pick_x, pick_y, pin_x, pin_y)!
            x_coord, y_coord, x_pin, y_pin = action
            x_coord = max(min(x_coord, high[0]), low[0])
            y_coord = max(min(y_coord, high[1]), low[1])
            # if pin point is OOB, don't pin (don't constrain x_pin/y_pin)
            #x_pin = max(min(x_pin, high[0]), low[0])
            #y_pin = max(min(y_pin, high[1]), low[1])
            # assume clip act space
            x_coord = (x_coord / 2.0) + 0.5
            y_coord = (y_coord / 2.0) + 0.5
            x_pin = (x_pin / 2.0) + 0.5
            y_pin = (y_pin / 2.0) + 0.5
            # dx/dy: direction is [x2-x1, y2-y1]; magnitude is the distance to the nearest edge
            dx = x_coord - x_pin
            if dx > 0:
                #dx_ = 1 + self._slack/4 - x_coord
                dx_ = 1 - x_coord
            else:
                #dx_ = -(x_coord + self._slack/4)
                dx_ = -x_coord
            if dx != 0:
                x_factor = dx_ / dx
            dy = y_coord - y_pin
            if dy > 0:
                #dy_ = 1 + self._slack/4 - y_coord
                dy_ = 1 - y_coord
            else:
                #dy_ = -(y_coord + self._slack/4)
                dy_ = -y_coord
            if dy != 0:
                y_factor = dy_ / dy
            if dx == 0 and dy == 0:
                factor = 0
            elif dx == 0:
                factor = y_factor
            elif dy == 0:
                factor = x_factor
            else:
                factor = min(x_factor, y_factor)
            delta_x, delta_y = dx * factor, dy * factor
            delta_x = max(min(delta_x, high[2]), low[2])
            delta_y = max(min(delta_y, high[3]), low[3])
            self.gripper.grab_top(x_pin, y_pin, bilateral=True) # do we want to only grab the top layer?
        elif self._delta_actions:
            x_coord, y_coord, delta_x, delta_y = action
            x_coord = max(min(x_coord, high[0]), low[0])
            y_coord = max(min(y_coord, high[1]), low[1])
            delta_x = max(min(delta_x, high[2]), low[2])
            delta_y = max(min(delta_y, high[3]), low[3])
        else:
            x_coord, y_coord, length, radians = action
            x_coord = max(min(x_coord, high[0]), low[0])
            y_coord = max(min(y_coord, high[1]), low[1])
            length  = max(min(length,  high[2]), low[2])
            r_trunc = max(min(radians, high[3]), low[3])

        if (not self._bilateral) and self._clip_act_space:
            # If we're here, then all four of these are in the range [-1,1].
            # Due to noise it might originally be out of range, but we truncated.
            x_coord = (x_coord / 2.0) + 0.5
            y_coord = (y_coord / 2.0) + 0.5
            if self._delta_actions:
                pass
            else:
                length = (length / 2.0) + 0.5
                r_trunc = r_trunc * np.pi
        # After this, we assume ranges {[0,1], [0,1],  [0,1], [-pi,pi]}.
        # Or if delta actions,         {[0,1], [0,1], [-1,1],   [-1,1]}.
        # Actually for non deltas, we have slack applied ...
        self.gripper.grab_top(x_coord, y_coord)
        
        # Hacky solution to make us forcibly grip.
        if self._force_grab == True:
            logger.debug('Inside force_grab, make sure we are EVALUATING!')
            radius_old = self.gripper.grip_radius
            while len(self.gripper.grabbed_pts) == 0:
                self.gripper.grip_radius += self._radius_inc
                logger.debug('radius incremented to: {}'.format(self.gripper.grip_radius))
                self.gripper.grab_top(x_coord, y_coord)
            if self.gripper.grip_radius != radius_old:
                self.gripper.grip_radius = radius_old
                logger.debug('radius reset to: {}'.format(self.gripper.grip_radius))
            logger.debug('done, grabbed points: {}'.format(self.gripper.grabbed_pts))

        # Determine direction on UNIT CIRCLE, then downscale by reduce_factor to
        # ensure we only move a limited amount each time step, might help physics?
        if self._delta_actions:
            total_length = np.sqrt( (delta_x)**2 + (delta_y)**2 )
            x_dir = delta_x / (total_length + _EPS)
            y_dir = delta_y / (total_length + _EPS)
        else:
            x_dir = np.cos(r_trunc)
            y_dir = np.sin(r_trunc)
        x_dir_r = x_dir * self.reduce_factor
        y_dir_r = y_dir * self.reduce_factor

        # Number of iterations for each stage of the action. Actually, the
        # iteration for the pull can be computed here ahead of time.
        if self._delta_actions:
            ii = 0
            current_l = 0
            while True:
                current_l += np.sqrt( (x_dir_r)**2 + (y_dir_r)**2 )
                if current_l >= total_length:
                    break
                ii += 1
            iters_pull = ii
        else:
            iters_pull = int(self.iters_pull_max * length)

        pull_up    = self.iters_up + self.iters_up_rest
        rest_start = self.iters_up + self.iters_up_rest + iters_pull
        drop_start = self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest
        iterations = self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest + self.iters_rest

        if initialize:
            logger.info("         ======== [during obs.reset()] EXECUTING ACTION: {} ========".format(astr))
        else:
            logger.info("         ======== EXECUTING ACTION: {} ========".format(astr))
        logger.debug("Gripped at ({:.2f}, {:.2f})".format(x_coord, y_coord))
        if self._bilateral:
            logger.debug("Pinned at ({:.2f}, {:.2f})".format(x_pin, y_pin))
            logger.debug("dx {:.2f}, dy {:.2f}".format(delta_x, delta_y))
            logger.debug("Pinned points: {}".format(self.gripper.pinned_pts))
        logger.debug("Grabbed points: {}".format(self.gripper.grabbed_pts))
        logger.debug("Total grabbed: {}".format(len(self.gripper.grabbed_pts)))
        logger.debug("Action maps to {:.3f}, {:.3f}".format(x_dir, y_dir))
        logger.debug("Actual magnitudes: {:.4f}, {:.4f}".format(x_dir_r, y_dir_r))
        logger.debug("itrs up / wait / pull / wait / drop+rest: {}, {}, {}, {}, {}".format(
            self.iters_up, self.iters_up_rest, iters_pull, self.iters_grip_rest, self.iters_rest))

        # Add special (but potentially common) case, if our gripper grips nothing.
        #if len(self.gripper.grabbed_pts) == 0 or (self._bilateral and len(self.gripper.pinned_pts) == 0):
        if len(self.gripper.grabbed_pts) == 0:
            logger.info("No points gripped! Exiting action ...")
            self.gripper.release()
            self.gripper.release(bilateral=True)
            exit_early = True
            iterations = 0

        i = 0
        # if self._bilateral:
        #     time.sleep(10)
        while i < iterations:
            self._pull(i, iters_pull, x_dir_r, y_dir_r)
            self.cloth.update()
            if not initialize:
                self.num_sim_steps += 1

            # Debugging -- move to separate method?
            if i == self.iters_up:
                logger.debug("i {}, now pulling".format(self.iters_up))
            elif i == drop_start:
                logger.debug("i {}, now dropping".format(drop_start))
            if self.debug_viz and i % 3 == 0:
                self._debug_viz_plots()

            # If we get any tears (pull in a bad direction, etc.), exit.
            if self.cloth.have_tear:
                if self._bilateral:
                    # Stop a bilateral action at the tear threshold but do not terminate episode
                    logger.debug("Tear threshold met, dropping cloth...")
                    self.gripper.release()
                    self.cloth.cloth_have_tear = False
                else:
                    logger.debug("TEAR, exiting...")
                    self.have_tear = True
                    break
            i += 1

        if initialize:
            return
        self.num_steps += 1
        rew  = self._reward(action, exit_early)
        term = self._terminal()
        self.logger.info("Reward: {:.4f}. Terminal: {}".format(rew, term))
        self.logger.info("Steps/SimSteps: {}, {}".format(self.num_steps, self.num_sim_steps))
        info = {
            'num_steps': self.num_steps,
            'num_sim_steps': self.num_sim_steps,
            'actual_coverage': self._current_coverage,
            'start_coverage': self._start_coverage,
            'variance_inv': self._compute_variance(),
            'start_variance_inv': self._start_variance_inv,
            'have_tear': self.have_tear,
            'out_of_bounds': self._out_of_bounds(),
        }
        return self.state, rew, term, info

    def _reward(self, action, exit_early):
        """Reward function.

        First we apply supporting and auxiliary rewards. Then we define actual
        coverage approximations. For now we are keeping the reward as deltas
        and then a large bonus for task completion.

        If we want to ignore any of the extra 'auxiliary' penalties/rewards,
        modify vaules in `self.__init__` instead of commenting out here.

        Parameters
        ----------
        action: The action taken from self.step()
        exit_early: True if the agent never gripped anything. Then we'll
            probably want to apply a penalty.
        """
        log = self.logger

        # Keep adjusting this to get cumulative reward.
        rew = 0

        # Apply one of tear OR oob penalities. Then one for bad / wasted actions.
        if self.have_tear:
            rew += self._tear_penalty
            log.debug("Apply tear penalty, reward {:.2f}".format(rew))
        elif self._out_of_bounds():
            rew += self._oob_penalty
            log.debug("Apply out of bounds penalty, reward {:.2f}".format(rew))
        if exit_early:
            rew += self._nogrip_penalty
            log.debug("Apply no grip penalty, reward {:.2f}".format(rew))

        # Apply penalty if action outside the bounds.
        def penalize_action(aval, low, high):
            if low <= aval <= high:
                return 0.0
            if aval < low:
                diff = low - aval
            else:
                diff = aval - high
            pen = - min(diff**2, self._act_pen_limit) * self._act_bound_factor
            assert pen < 0
            return pen

        if not self._clip_act_space:
            x_coord, y_coord, length, radians = action
            low = self.action_space.low
            high = self.action_space.high
            pen0 = penalize_action(x_coord, low[0], high[0])
            pen1 = penalize_action(y_coord, low[1], high[1])
            pen2 = penalize_action(length,  low[2], high[2])
            pen3 = penalize_action(radians, low[3], high[3])
            rew += pen0
            rew += pen1
            rew += pen2
            rew += pen3
            log.debug("After action pen. {:.2f} {:.2f} {:.2f} {:.2f}, rew {:.2f}".
                    format(pen0, pen1, pen2, pen3, rew))

        # Define several coverage formulas. Subtle point about deltas: the
        # input is reward but not the 'auxiliary' bonuses, etc.

        def _save_bad_hull(points, eps=1e-4):
            log.warn("Bad ConvexHull hull! Note, here len(points): {}".format(len(points)))
            pth_head = 'bad_cloth_hulls'
            if not os.path.exists(pth_head):
                os.makedirs(pth_head, exist_ok=True)
            num = len([x for x in os.listdir(pth_head) if 'cloth_' in x])
            if self._logger_idx is not None:
                cloth_file = join(pth_head,
                        'cloth_{}_subrank_{}.pkl'.format(num+1, self._logger_idx))
            else:
                cloth_file = join(pth_head, 'cloth_{}.pkl'.format(num+1))
            self.save_state(cloth_file)

        def compute_height():
            threshold = self.cfg['cloth']['thickness'] / 2.0
            allpts = self.cloth.allpts_arr  # fyi, includes pinned
            z_vals = allpts[:,2]
            num_below_thresh = np.sum(z_vals < threshold)
            fraction = num_below_thresh / float(len(z_vals))
            return fraction

        def compute_variance():
            allpts = self.cloth.allpts_arr
            z_vals = allpts[:,2]
            variance = np.var(z_vals)
            if variance < 0.000001: # handle asymptotic behavior
                return 1000
            else:
                return 0.001 / variance

        def compute_coverage():
            points = np.array([[min(max(p.x,0),1), min(max(p.y,0),1)] for p in self.cloth.pts])
            try:
                # In 2D, this actually returns *AREA* (hull.area returns perimeter)
                hull = ConvexHull(points)
                coverage = hull.volume
            except scipy.spatial.qhull.QhullError as e:
                logging.exception(e)
                _save_bad_hull(points)
                coverage = 0
            return coverage

        def compute_delta(reward):
            diff = reward - self._prev_reward
            self._prev_reward = reward
            return diff

        # Huge reward if we've succeeded in coverage.  This is where we assign
        # to _current_coverage, so be careful if this is what we want, e.g., if
        # we've torn the cloth or are out of bounds, it's not updated.
        self._current_coverage = compute_coverage()
        if self._current_coverage > _REWARD_THRESHOLDS['coverage']:
            rew += self._cover_success
            log.debug("Success in coverage!! reward {:.2f}".format(rew))

        rew += self._neg_living_rew
        log.debug("After small living penalty, reward {:.2f}".format(rew))

        # August 2, 2019: only using coverage and coverage-delta.
        if self.reward_type == 'coverage':
            # Actual coverage of the cloth on the XY-plane.
            rew += compute_coverage()
        elif self.reward_type == 'coverage-delta':
            # * difference * in coverage on the XY-plane.
            rew += compute_delta(compute_coverage())
        elif self.reward_type == 'height':
            # Proportion of points that are below thickness-dependent threshold.
            rew += compute_height()
        elif self.reward_type == 'height-delta':
            # * difference * in the proportion of points below threshold.
            rew += compute_delta(compute_height())
        elif self.reward_type == 'variance':
            # 1/variance in Z-coordinate (to punish high variance)
            rew += compute_variance()
        elif self.reward_type == 'variance-delta':
            # * difference * in 1/variance.
            rew += compute_delta(compute_variance())
        elif self.reward_type == 'folding-number':
            # Probably never doing this anytime soon. :-)
            raise NotImplementedError()
        else:
            raise ValueError(self.reward_type)

        log.debug("COVERAGE {:.2f}, reward at end {:.2f}".format(self._current_coverage, rew))
        return rew

    def _terminal(self):
        """Detect if we're done with an episode, for any reason.

        First we detect for (a) exceeding max steps, (b) tearing, (c) out of
        bounds. Then we check if we have sufficiently covered the plane.
        """
        done = False

        if self.num_steps >= self.max_actions:
            self.logger.info("num_steps {} >= max_actions {}, hence done".format(
                    self.num_steps, self.max_actions))
            done = True
        elif self.have_tear:
            self.logger.info("A \'tear\' exists, hence done")
            done = True
        elif self._out_of_bounds():
            self.logger.info("Went out of bounds, hence done")
            done = True

        # Assumes _current_coverage set in _reward() call before _terminal().
        assert 'coverage' in self.reward_type
        rew_thresh = _REWARD_THRESHOLDS[self.reward_type]
        if self._current_coverage > rew_thresh:
            self.logger.info("Cloth is sufficiently smooth, {:.3f} exceeds "
                "threshold {:.3f} hence done".format(self._prev_reward, rew_thresh))
            done = True

        if done and self.render_gl:
            #self.render_proc.terminate()
            self.cloth.stop_render()
            #self.render_proc = None
        return done

    def reset(self):
        """Must call each time we start a new episode.

        Initializes to a new state, depending on the init 'type' in the config.

        `self.num_steps`: number of actions or timesteps in an episode.
        `self.num_sim_steps`: number of times we call `cloth.update()`.

        The above don't count any of the 'initialization' actions or steps --
        only those in the actual episode.

        Parameters
        ----------
        state: {"pts": [list of Points], "springs": [list of Springs]}
            If specified, load the cloth with this specific state and skip initialization.
        """
        reset_start = time.time()
        logger = self.logger
        cfg = self.cfg
        if self._start_state:
            self.cloth = cloth = Cloth(params=self.cfg,
                                       render=self.render_gl,
                                       random_state=self.np_random,
                                       render_port=self.render_port,
                                       state=copy.deepcopy(self._start_state))
        else:
            self.cloth = cloth = Cloth(params=self.cfg,
                                       render=self.render_gl,
                                       random_state=self.np_random,
                                       render_port=self.render_port)
        assert len(cloth.pts) == self.num_points, \
                "{} vs {}".format(len(cloth.pts), self.num_points)
        assert cloth.bounds[0] == self.bounds[0]
        assert cloth.bounds[1] == self.bounds[1]
        assert cloth.bounds[2] == self.bounds[2]
        self.gripper = gripper = Gripper(cloth, self.grip_radius,
                self.cfg['cloth']['height'], self.cfg['cloth']['thickness'])
        self.num_steps = 0
        self.num_sim_steps = 0
        self.have_tear = False

        if self.debug_viz:
            self.logger.info("Note: we set our config to visualize the init. We"
                    " will now play a video ...")
            nrows, ncols = 1, 2
            self.plt = plt
            self.debug_fig = plt.figure(figsize=(12*ncols,12*nrows))
            self.debug_ax1 = self.debug_fig.add_subplot(1, 2, 1)
            self.debug_ax2 = self.debug_fig.add_subplot(1, 2, 2, projection='3d')
            self.debug_ax2.view_init(elev=5., azim=-50.)
            self.plt.ion()
            self.plt.tight_layout()

        # Handle starting states, assuming we don't already have one.
        if not self._start_state:
            self._reset_actions()

        reset_time = (time.time() - reset_start) / 60.0
        logger.debug("Done with initial state, {:.2f} minutes".format(reset_time))

        # Adding to ensure prev_reward is set correctly after init, if we are
        # using deltas. Assign here because it's after we load/init the cloth.
        assert 'coverage' in self.reward_type
        self._prev_reward = self._compute_coverage()
        self._start_coverage = self._prev_reward
        self._start_variance_inv = self._compute_variance()

        # We shouldn't need to wrap around np.array(...) as self.state does that.
        # Ryan: compute dom rand params once per episode
        self.dom_rand_params['gval_depth'] = self.np_random.uniform(low=40, high=50) # really pixels ...
        self.dom_rand_params['gval_rgb'] = self.np_random.uniform(low=0.7, high=1.3)
        lim = self.np_random.uniform(low=-15.0, high=15.0)
        #self.dom_rand_params['noise'] = self.np_random.uniform(low=-lim, high=lim, size=(self._wd, self._hd, 3))
        self.dom_rand_params['c'] = np.random.uniform(low=0.4, high=0.6, size=(3,))
        self.dom_rand_params['n1'] = np.random.uniform(low=-0.35, high=0.35, size=(3,))
        self.dom_rand_params['camera_pos'] = np.random.normal(0., scale=0.04, size=(3,)) # check get_image_rep_279.py for 'scale'
        self.dom_rand_params['camera_deg'] = np.random.normal(0., scale=0.90, size=(3,))
        self.dom_rand_params['specular_max'] = np.random.uniform(low=0.0, high=0.0) # check get_image_rep_279.py for 'high'
        obs = self.state
        return obs

    def _reset_actions(self):
        """Helper for reset in case reset applies action.

        Mainly to help clean up the code. Note that cfg['init']['type'] is
        heavily intertwined with the cloth.pyx initialization, so always
        cross-reference with code there. If we decide on fixing how states are
        initialized, do it here (and with cloth.pyx if needed).

        Remember that actions applied to self.step need to be consistent with
        whether we're using delta or non-delta actions. Within self.step, it
        gets converted to the non-delta if needed, but if our _settings_ say we
        use delta, we need delta actions. Similarly for clipping actions, our
        step will automatically 'unclip' these.

        Use self.np_random for anything random-related, NOT np.random.

        See comments inside the code for specifics regarding the tiers.

        To get the highest point, use:
            highest_point = max(self.cloth.pts, key=lambda pt: pt.z)
        Or a list in descending order:
            high_points = sorted(self.cloth.pts, key=lambda pt: pt.z, reverse=True)
        """
        logger = self.logger
        cfg = self.cfg
        init_side = 1 if self.cloth.init_side else -1
        old_bilateral = self._bilateral 
        self._bilateral = False # we do not want bilateral actions during reset()

        def _randval_minabs(low, high, minabs=None):
            # Helper to handle ranges with minabs requirements.
            val = self.np_random.uniform(low=low, high=high)
            if minabs is not None:
                assert minabs > 0, minabs
                assert low < -minabs or high > minabs
                while np.abs(val) < minabs:
                    val = self.np_random.uniform(low=low, high=high)
            return val

        def _prevent_oob(val, dval, lower=0.0, upper=1.0):
            # Helper to somewhat mitigate prevent out of bounds
            if val + dval < lower:
                dval = lower - val
            elif val + dval > upper:
                dval = upper - val
            return dval

        if self._init_type == 'tier1':
            # ------------------------------------------------------------------
            # Should reveal all corners, or be reasonably easy. Ideally, don't
            # pull the cloth out of bounds by too much. I do this with two
            # (optionally three, if the first two didn't do much) pulls
            # starting from a flat cloth, where the two pulls are relatively
            # short and attempt not to go out of bounds too much.
            # ------------------------------------------------------------------
            lim = 0.20

            # First pull.
            p0 = self.cloth.pts[ self.np_random.randint(len(self.cloth.pts)) ]
            if self._delta_actions:
                dx0 = _randval_minabs(low=-lim, high=lim, minabs=0.08)
                dy0 = _randval_minabs(low=-lim, high=lim, minabs=0.08)
                dx0 = _prevent_oob(p0.x, dx0)
                dy0 = _prevent_oob(p0.y, dy0)
                action0 = (p0.x, p0.y, dx0, dy0)
            else:
                raise NotImplementedError()
            action0 = self._convert_action_to_clip_space(action0)
            self.step(action0, initialize=True)

            # Second pull.
            p1 = self.cloth.pts[ self.np_random.randint(len(self.cloth.pts)) ]
            if self._delta_actions:
                dx1 = _randval_minabs(low=-lim, high=lim, minabs=0.08)
                dy1 = _randval_minabs(low=-lim, high=lim, minabs=0.08)
                dx1 = _prevent_oob(p1.x, dx1)
                dy1 = _prevent_oob(p1.y, dy1)
                action1 = (p1.x, p1.y, dx1, dy1)
            else:
                raise NotImplementedError()
            action1 = self._convert_action_to_clip_space(action1)
            self.step(action1, initialize=True)

            # Third pull if the cloth looks too flat.
            if self._compute_coverage() >= 0.90:
                p2 = self.cloth.pts[ self.np_random.randint(len(self.cloth.pts)) ]
                if self._delta_actions:
                    dx2 = _randval_minabs(low=-lim, high=lim, minabs=0.08)
                    dy2 = _randval_minabs(low=-lim, high=lim, minabs=0.08)
                    dx2 = _prevent_oob(p2.x, dx2)
                    dy2 = _prevent_oob(p2.y, dy2)
                    action2 = (p2.x, p2.y, dx2, dy2)
                else:
                    raise NotImplementedError()
                action2 = self._convert_action_to_clip_space(action2)
                self.step(action2, initialize=True)

        elif self._init_type == 'tier2':
            # ------------------------------------------------------------------
            # Should have some difficulty (e.g., one corner hidden). The
            # cloth.pyx randomly picks which side (init_side) to let the cloth
            # drop from. The cloth settles, and then we pick one of the two top
            # most corners, and pull it towards the center of the frame. Then
            # we make a second pull that tries to cover it. This setup could
            # even double as our 'bed-making task' if we wanted one.
            # ------------------------------------------------------------------
            for i in range(1500):
                self.cloth.update()
            logger.debug("Cloth settled, now apply actions.")

            # Pick one of the two topmost corners ('top' -1, 'bottom' -25).
            idx = -25 if self.np_random.rand() < 0.5 else -1

            # Apply first pull, roughly towards the center (i.e., don't have
            # high dx0, make dy0 positive if bottom, reverse if not, etc).
            p0 = self.cloth.pts[ idx ]
            if self._delta_actions:
                dx0 = self.np_random.uniform(0.30, 0.50) * init_side
                if idx == -25:
                    dy0 = self.np_random.uniform(0.30, 0.60)
                else:
                    dy0 = self.np_random.uniform(-0.60, -0.30)
                action0 = (p0.x, p0.y, dx0, dy0)
            else:
                raise NotImplementedError()
            action0 = self._convert_action_to_clip_space(action0)
            self.step(action0, initialize=True)

            # Now attempt to cover that corner. Here we can have a slightly
            # longer range of dx0, also make dy0 now reverse of earlier, but
            # dx0 has same magnitude as we pull in same general x direction.
            if idx == -25:
                p1 = self.cloth.pts[-19]
                if self._delta_actions:
                    dx1 = self.np_random.uniform(0.30, 0.60) * init_side
                    dy1 = self.np_random.uniform(-0.30, -0.60)
                    action1 = (p1.x, p1.y, dx1, dy1)
                else:
                    raise NotImplementedError()
                action1 = self._convert_action_to_clip_space(action1)
            elif idx == -1:
                p1 = self.cloth.pts[-7]
                if self._delta_actions:
                    dx1 = self.np_random.uniform(0.30, 0.60) * init_side
                    dy1 = self.np_random.uniform(0.30, 0.60)
                    action1 = (p1.x, p1.y, dx1, dy1)
                else:
                    raise NotImplementedError()
                action1 = self._convert_action_to_clip_space(action1)
            self.step(action1, initialize=True)

            logger.debug("Let's continue simulating to finish the reset().")
            for i in range(500):
                self.cloth.update()

        elif self._init_type == 'tier3':
            # ------------------------------------------------------------------
            # Should be the hardest we consider, perhaps all corners hidden.
            # From a flat cloth (so no need for extra settling iterations) why
            # not do a very high (random) pull? See how that works. Also since
            # it's a flat cloth we can literally choose the (x,y) instead of
            # picking a point at random.
            # ------------------------------------------------------------------
            old_iters_up = self.iters_up
            self.iters_up = self.np_random.uniform(low=200, high=280)
            logger.debug("Setting self.iters_up: {}.".format(self.iters_up))
            lim = 0.25

            # A high pull.
            if self._delta_actions:
                p0x = _randval_minabs(low=0.30, high=0.70)
                p0y = _randval_minabs(low=0.30, high=0.70)
                dx0 = _randval_minabs(low=-lim, high=lim, minabs=0.10)
                dy0 = _randval_minabs(low=-lim, high=lim, minabs=0.10)
                dx0 = _prevent_oob(p0x, dx0)
                dy0 = _prevent_oob(p0y, dy0)
                action0 = (p0x, p0y, dx0, dy0)
            else:
                raise NotImplementedError()
            action0 = self._convert_action_to_clip_space(action0)
            self.step(action0, initialize=True)

            # Settle, and then re-assign to `self.iters_up`.
            logger.debug("Let's continue simulating to finish the reset().")
            for i in range(800):
                self.cloth.update()
            self.iters_up = old_iters_up
        else:
            raise ValueError(self._init_type)

        logger.debug("STARTING COVERAGE: {:.2f}".format(self._compute_coverage()))
        logger.debug("STARTING VARIANCE: {:.2f}".format(self._compute_variance()))
        self._bilateral = old_bilateral

    def get_random_action(self, atype='over_xy_plane'):
        """Retrieves random action.

        One way is to use the usual sample method from gym action spaces. Since
        we set the cloth plane to be in the range (0,1) in the x and y
        directions by default, we will only sample points over that range. This
        may or may not be desirable; we will sometimes pick points that don't
        touch any part of the cloth, in which case we just do a 'NO-OP'.

        The other option would be to sample any point that touches the cloth, by
        randomly picking a point from the cloth mesh and then extracting its x
        and y. We thus always touch something, via the 'naive cylinder' method.
        Though right now we don't support the delta actions here, for some reason.
        """
        if atype == 'over_xy_plane':
            return self.action_space.sample()
        elif atype == 'touch_cloth':
            assert not self._delta_actions
            pt = self.cloth.pts[ self.np_random.randint(self.num_points) ]
            length = self.np_random.uniform(low=0, high=1)
            angle = self.np_random.uniform(low=-np.pi, high=np.pi)
            action = (pt.x, pt.y, length, angle)
            if self._clip_act_space:
                action = ((pt.x - 0.5) * 2,
                          (pt.y - 0.5) * 2,
                          (length - 0.5) * 2,
                          angle / np.pi)
            return action
        else:
            raise ValueError(atype)

    def _out_of_bounds(self):
        """Detect if we're out of bounds, e.g., to stop an action.

        Currently, bounds are [0,1]. We add some slack for x/y bounds to
        represent cloth that drapes off the edge of the bed.  We should not be
        able to grasp these points, however.
        """
        pts = self.cloth.allpts_arr
        ptsx = pts[:,0]
        ptsy = pts[:,1]
        ptsz = pts[:,2]
        cond1 = np.max(ptsx) >= self.cloth.bounds[0] + self._slack
        cond2 = np.min(ptsx) < - self._slack
        cond3 = np.max(ptsy) >= self.cloth.bounds[1] + self._slack
        cond4 = np.min(ptsy) < - self._slack
        cond5 = np.max(ptsz) >= self.cloth.bounds[2]
        cond6 = np.min(ptsz) < 0
        outb = (cond1 or cond2 or cond3 or cond4 or cond5 or cond6)
        if outb:
           self.logger.debug("np.max(ptsx): {:.4f},  cond {}".format(np.max(ptsx), cond1))
           self.logger.debug("np.min(ptsx): {:.4f},  cond {}".format(np.min(ptsx), cond2))
           self.logger.debug("np.max(ptsy): {:.4f},  cond {}".format(np.max(ptsy), cond3))
           self.logger.debug("np.min(ptsy): {:.4f},  cond {}".format(np.min(ptsy), cond4))
           self.logger.debug("np.max(ptsz): {:.4f},  cond {}".format(np.max(ptsz), cond5))
           self.logger.debug("np.min(ptsz): {:.4f},  cond {}".format(np.min(ptsz), cond6))
        return outb

    def render(self, filepath, mode='human', close=False):
        """Much subject to change.

        If mode != 'matplotlib', spawn a child process rendering the cloth.
        As a result, you only need to call this once rather than every time
        step. The process is terminated with terminal(), so you must call
        it again after each episode, before calling reset().

        You will have to pass in the renderer filepath to this program, as the
        package will be unable to find it. To get the filepath from, for example,
        gym-cloth/examples/[script_name].py, run

        >>> this_dir = os.path.dirname(os.path.realpath(__file__))
        >>> filepath = os.path.join(this_dir, "../render/build")
        """
        if mode == 'matplotlib':
            self._debug_viz_plots()
        elif self.render_gl and not self.render_proc:
            owd = os.getcwd()
            os.chdir(filepath)
            dev_null = open('/dev/null','w')
            self.render_proc = subprocess.Popen(["./clothsim"], stdout=dev_null, stderr=dev_null)
            os.chdir(owd)

    # --------------------------------------------------------------------------
    # Random helper methods, debugging, etc.
    # --------------------------------------------------------------------------

    def _compute_variance(self):
        """Might want to use this instead of the internal method in reward()?
        """
        allpts = self.cloth.allpts_arr
        z_vals = allpts[:,2]
        variance = np.var(z_vals)
        if variance < 0.000001: # handle asymptotic behavior
            return 1000
        else:
            return 0.001 / variance

    def _compute_coverage(self):
        """Might want to use this instead of the internal method in _reward()?
        """
        points = np.array([[min(max(p.x,0),1), min(max(p.y,0),1)] for p in self.cloth.pts])
        try:
            # In 2D, this actually returns *AREA* (hull.area returns perimeter)
            hull = ConvexHull(points)
            coverage = hull.volume
        except scipy.spatial.qhull.QhullError as e:
            logging.exception(e)
            #_save_bad_hull(points)
            coverage = 0
        return coverage

    def _debug_viz_plots(self):
        """Use `plt.ion()` for interactive plots, requires `plt.pause(...)` later.

        This is for the debugging part of the initialization process. It's not
        currently meant for the actual rendering via `env.render()`.
        """
        plt = self.plt
        ax1 = self.debug_ax1
        ax2 = self.debug_ax2
        eps = 0.05

        ax1.cla()
        ax2.cla()
        pts  = self.cloth.noncolorpts_arr
        cpts = self.cloth.colorpts_arr
        ppts = self.cloth.pinnedpts_arr
        if len(pts) > 0:
            ax1.scatter(pts[:,0], pts[:,1], c='g')
            ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c='g')
        if len(cpts) > 0:
            ax1.scatter(cpts[:,0], cpts[:,1], c='b')
            ax2.scatter(cpts[:,0], cpts[:,1], cpts[:,2], c='b')
        if len(ppts) > 0:
            ax1.scatter(ppts[:,0], ppts[:,1], c='darkred')
            ax2.scatter(ppts[:,0], ppts[:,1], ppts[:,2], c='darkred')
        ax1.set_xlim([0-eps, 1+eps])
        ax1.set_ylim([0-eps, 1+eps])
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_zlim([0, 1])
        plt.pause(0.0001)

    def _save_matplotlib_img(self, target_dir=None):
        """Save matplotlib image into a target directory.
        """
        #Adi: Quick test
        print("SAVING AN IMAGE!!!")
        plt = self.plt
        if target_dir is None:
            target_dir = (self.fname_log).replace('.log','.png')
        print("Note: saving matplotlib img of env at {}".format(target_dir))
        plt.savefig(target_dir)

    def _setup_logger(self):
        """Set up the logger (and also save the config w/similar name).

        If you create a new instance of the environment class in the same
        program, you will get duplicate logging messages. We should figure out a
        way to fix that in case we want to scale up to multiple environments.

        Daniel TODO: this is going to refer to the root logger and multiple
        instantiations of the environment class will result in duplicate
        logging messages to stdout. It's harmless wrt environment stepping.
        """
        cfg = self.cfg
        dstr = '-{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        filename = (cfg['log']['file']).replace('.log',dstr)
        if self._logger_idx is not None:
            filename = filename.replace('.log',
                    '_rank_{}.log'.format(str(self._logger_idx).zfill(2)))
        logging.basicConfig(
                level=_logging_setup_table[cfg['log']['level']],
                filename=filename,
                filemode='w')

        # Define a Handler which writes messages to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(_logging_setup_table[cfg['log']['level']])

        # Set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s '
                                      '%(message)s', datefmt='%m-%d %H:%M:%S')

        # Tell the handler to use this format, and add handler to root logger
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        if self._logger_idx is not None:
            self.logger = logging.getLogger("cloth_env_{}".format(self._logger_idx))
        else:
            self.logger = logging.getLogger("cloth_env")

        # Finally, save config file so we can exactly reproduce parameters.
        json_str = filename.replace('.log','.json')
        with open(json_str, 'w') as fh:
            json.dump(cfg, fh, indent=4, sort_keys=True)
        self.fname_log = filename
        self.fname_json = json_str

    def _act2str(self, action):
        """Turn an action into something more human-readable.
        """
        if self._delta_actions:
            x, y, dx, dy = action
            astr = "({:.2f}, {:.2f}), deltax {:.2f}, deltay {:.2f}".format(
                    x, y, float(dx), float(dy))
        else:
            x, y, length, direction = action
            if self._clip_act_space:
                astr = "({:.2f}, {:.2f}), length {:.2f}, angle {:.2f}".format(
                    x, y, float(length), float(direction))
                astr += "  Re-scaled: ({:.2f}, {:.2f}), {:.2f}, {:.2f}".format(
                    (x/2)+0.5, (y/2)+0.5, (length/2)+0.5, direction*np.pi)
            else:
                astr = "({:.2f}, {:.2f}), length {:.2f}, angle {:.2f}".format(
                    x, y, float(length), float(direction))
        return astr

    def _convert_action_to_clip_space(self, a):
        # Help out with all the clipping and stuff.
        if not self._clip_act_space:
            return a
        if self._delta_actions:
            newa = ((a[0]-0.5)*2, (a[1]-0.5)*2,         a[2],       a[3])
        else:
            newa = ((a[0]-0.5)*2, (a[1]-0.5)*2, (a[2]-0.5)*2, a[3]/np.pi)
        return newa

    def _adjust_gamma(self, image, gamma=1.0):
        """For darkening images.

        Builds a lookup table mapping the pixel values [0, 255] to their
        adjusted gamma values.
        https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 \
                for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
