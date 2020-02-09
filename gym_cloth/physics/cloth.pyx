# cython: profile=True
"""
A cloth class, consists of a collection of points and their corresponding
constraints.  Uses the springs class here as well.
"""
from gym_cloth.physics.point import Point
from gym_cloth.physics.gripper import Gripper
import numpy as np
import os
import math
import yaml
import sys
import zmq
from gym.utils import seeding


cdef inline double fastnorm(double x, double y, double z):
    return math.sqrt(x*x + y*y + z*z)


class Cloth(object):

    def __init__(self,
                 gravity=-9.8,
                 bounds=(1,1,1),
                 minimum_z=0,
                 params=None,
                 render=False,
                 render_port=5556,
                 random_state=None,
                 state=None):
        """
        Creates a cloth with width x height points spaced dx and dy apart.
        Points can be pinned.

        Parameters
        ----------
        random_state: np.random.RandomState
            Use this for better seeding. Otherwise, we can just use the normal
            params seed, but using the random state means each time we reset the
            environment, we can get different states, which is what we want. See
            `ClothEnv` class for further details.
        state: {"pts": [list of Points], "springs": [list of Springs]}
            If specified, start the cloth with `state` as its `self.pts`.
        """
        # Do NOT do this! If we change the config during code runs, we change
        # settings when this gets called during env.reset().
        #with open(fname) as fh:
        #    params = yaml.safe_load(fh)
        self.params = params
        # Adjust dx and dy so scale is [0,1]. Also see:
        # https://github.com/BerkeleyAutomation/gym-cloth/issues/29
        self.width  = width  = self.params["cloth"]["num_width_points"]
        self.height = height = self.params["cloth"]["num_height_points"]
        self.dx = dx = self.params["cloth"]["width"] * 1.0 / (width - 1)
        self.dy = dy = self.params["cloth"]["height"] * 1.0 / (height - 1)
        self.pts = []
        self.color_pts = []
        self.springs = []
        self.map = {}
        self.bounds = bounds
        self.minimum_z = minimum_z
        self.gravity = gravity
        self.render = render
        self.render_port = render_port
        self.cloth_have_tear = False
        if random_state is not None:
            self.np_random = random_state
        else:
            print('WARNING: we are creating a new np_random object with seed: {}'.format(
                    self.params['seed']))
            print('Normally, we want to be passing in an already created np_random.')
            self.np_random, _ = seeding.np_random(self.params['seed'])
        self.init_type = init_type = self.params['init']['type']
        self.init_side = (self.np_random.rand() > 0.5)

        # Older code for matplotlib plots, but we may end up pinning cloth anyway.
        pin_cond  = self.params['cloth']['pin_cond']
        color_pts = self.params['cloth']['color_pts']
        if pin_cond == "y=0":
            pin_cond = lambda x, y, height, width: y == 0
        elif pin_cond == "x=0,y=0" or pin_cond == "y=0,x=0" or pin_cond == "default":
            pin_cond = lambda x, y, height, width: x == 0 or y == 0
        else:
            raise ValueError(pin_cond)

        if state:
            self.pts = state["pts"]
            self.springs = state["springs"]
        else:
            assert height == width # For now
            for r in range(height):
                for c in range(width):
                    if init_type == 'tier2':
                        # ------------------------------------------------------
                        # Apply semi-random vertical drop.  Can be on either side of
                        # the plane. We record the side because some of the analytic
                        # policies need to know the specific indices of corners. If it's
                        # turning too complicated, we can always force init_side=True?
                        # ------------------------------------------------------
                        noise = self.np_random.rand() * 0.01 - 0.005
                        if r == 0:
                            noise = 0
                        if self.init_side:
                            xval = 0.0 + np.abs(noise)
                        else:
                            xval = 1.0 - np.abs(noise)
                        pt = Point(x = xval,
                                   y = dx*c,
                                   z = dy*r,
                                   boundsx=bounds[0],
                                   boundsy=bounds[1],
                                   boundsz=bounds[2],
                                   identity_0=r,
                                   identity_1=c)
                        self.pts.append(pt)
                    elif init_type == 'tier1' or init_type == 'tier3':
                        # ------------------------------------------------------
                        # Using r for x to make x stay fixed, y change more often. Should
                        # be OK, results in 184 render making the top surface brighter.
                        # ------------------------------------------------------
                        pt = Point(x = dx*r,
                                   y = dy*c,
                                   z = 0.0,
                                   boundsx=bounds[0],
                                   boundsy=bounds[1],
                                   boundsz=bounds[2],
                                   identity_0=r,
                                   identity_1=c)
                        self.pts.append(pt)
                    else:
                        raise ValueError(init_type)

                    # Add springs (i.e., constraints).
                    if r > 0:
                        self.springs.append(Spring(self.pts[(r - 1)*width + c], pt, "STRUCTURAL"))
                    if c > 0:
                        self.springs.append(Spring(self.pts[r*width + c - 1], pt, "STRUCTURAL"))
                    if r > 0 and c > 0:
                        self.springs.append(Spring(self.pts[(r - 1)*width + c - 1], pt, "SHEARING"))
                    if r > 0 and c + 1 < width:
                        self.springs.append(Spring(self.pts[(r - 1)*width + c + 1], pt, "SHEARING"))
                    if r > 1:
                        self.springs.append(Spring(self.pts[(r - 2)*width + c], pt, "BENDING"))
                    if c > 1:
                        self.springs.append(Spring(self.pts[r*width + c - 2], pt, "BENDING"))

                    # For debugging, often useful to color some points.
                    if color_pts == 'None':
                        pass
                    elif color_pts == 'circle0':
                        if abs((pt.x - 1.0)**2 + (pt.y - 1.0)** 2 - 0.05**2) < 0.10:
                            self.color_pts.append(pt)
                    elif color_pts == 'diag0':
                        if abs(pt.x - pt.y) < 0.05:
                            self.color_pts.append(pt)
                    elif color_pts == 'diag1':
                        if abs((1.0 - pt.x) - pt.y) < 0.05:
                            self.color_pts.append(pt)
                    else:
                        raise ValueError(color_pts)

        # Make self.color_pts a set b/c we want efficient membership checks.
        self.color_pts = set(self.color_pts)
        if render:
            self.init_render()
        self.iter = 0

    def update(self):
        """Calls one update.

        See 184 code for details on how these are implemented. For now we `cdef`
        them here because you can't do a cdef on variables from `self`.
        """
        cp = self.params["cloth"]
        cdef int frames_per_sec   = self.params['frames_per_sec']
        cdef int simulation_steps = self.params['simulation_steps']
        cdef double mass          = cp["density"] / self.width / self.height
        cdef double mass_times_g  = mass * self.gravity
        cdef double delta_t       = 1.0 / frames_per_sec / simulation_steps
        cdef double cp_ks         = cp["ks"]
        cdef double cp_damping    = cp['damping']
        cdef double thickness     = self.params["cloth"]["thickness"]
        cdef double p_friction    = self.params["cloth"]["plane_friction"]
        cdef double surface_off   = 0.0001
        cdef double tear_thresh   = self.params['cloth']['tear_thresh']

        # reset forces and add gravity
        self._reset_gravity(mass_times_g)

        # Hooke's law
        self._hookes(cp_ks)

        # Verlet integration
        self._verlet(mass, delta_t, cp_damping)

        # handle self-collision
        self.build_spatial_map()
        for pt in self.pts:
            self.self_collide(pt, simulation_steps, thickness)

        # handle collision with plane
        for pt in self.pts:
            self._handle_plane_collision(pt, p_friction, surface_off)

        # limit spring length changes to 10% per timestep (as per CS 184 project)
        self._limit_spring_changes(tear_thresh)

        if self.render and self.iter % 3 == 0: # can be tuned for "frame skip"
            state = dict()
            for pt in self.pts:
                state["%d %d" % (pt.identity_0, pt.identity_1)] = (pt.x, pt.y, pt.z)
            self.socket.send_json(state)
        self.iter += 1

    def _reset_gravity(self, double mass_times_g):
        for pt in self.pts:
            pt.reset_force_vector()
            pt.add_force(0, 0, mass_times_g)

    def _hookes(self, double cp_ks):
        cdef double KS_CORRECTION_CONSTANT

        for spring in self.springs:
            if spring.type == "BENDING":
                KS_CORRECTION_CONSTANT = 0.2
            else:
                KS_CORRECTION_CONSTANT = 1.0
            pa           = spring.ptA
            pb           = spring.ptB
            l2_norm_ab   = fastnorm(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z)
            force_mg     = cp_ks * KS_CORRECTION_CONSTANT * (l2_norm_ab - spring.rest_length) / l2_norm_ab
            force_on_a_0 = force_mg * (pb.x - pa.x)
            force_on_a_1 = force_mg * (pb.y - pa.y)
            force_on_a_2 = force_mg * (pb.z - pa.z)
            spring.ptA.add_force( force_on_a_0,  force_on_a_1,  force_on_a_2)
            spring.ptB.add_force(-force_on_a_0, -force_on_a_1, -force_on_a_2)

    def _verlet(self, double mass, double delta_t, double cp_damping):
        cdef double delta_sq_div_mass = (delta_t * delta_t) / mass
        cdef double damping           = (1.0 - cp_damping / 100.0)

        for pt in self.pts:
            if pt.pinned:
                continue
            cur_p_x = pt.x
            cur_p_y = pt.y
            cur_p_z = pt.z
            new_p_x = pt.x + (damping * (pt.x - pt.px)) + (pt.fx * delta_sq_div_mass)
            new_p_y = pt.y + (damping * (pt.y - pt.py)) + (pt.fy * delta_sq_div_mass)
            new_p_z = pt.z + (damping * (pt.z - pt.pz)) + (pt.fz * delta_sq_div_mass)
            # Actually, we'll handle this in the cloth-plane collision!  By forcing
            # the 'min' here, we actually make the cloth 'smooth' itself undesirably.
            #new_p_z = max(new_p_z, self.minimum_z)
            pt.set_position_vec(     new_p_x, new_p_y, new_p_z)
            pt.set_last_position_vec(cur_p_x, cur_p_y, cur_p_z)

    def _limit_spring_changes(self, double tear_thresh):
        """As per CS 184 class, limit changes.

        Also, here we'll detect tears, since we're already going through the for
        loop to calculate distances among points for each constraint. If we get
        a tear, terminate the associated environment episode.
        """
        for s in self.springs:
            pa = s.ptA
            pb = s.ptB
            if pa.pinned and pb.pinned:
                continue
            cur_length = fastnorm(pa.x - pb.x, pa.y - pb.y, pa.z - pb.z)

            if (cur_length > s.rest_length * tear_thresh):
                self.cloth_have_tear = True

            if (cur_length > (s.rest_length * 1.1)):
                dir_ba_x   = (pa.x - pb.x) / cur_length
                dir_ba_y   = (pa.y - pb.y) / cur_length
                dir_ba_z   = (pa.z - pb.z) / cur_length
                extra_diff = cur_length - s.rest_length * 1.1

                if pa.pinned:
                    pb.set_position_vec(pb.x + dir_ba_x * extra_diff,
                                        pb.y + dir_ba_y * extra_diff,
                                        pb.z + dir_ba_z * extra_diff)
                elif pb.pinned:
                    pa.set_position_vec(pa.x - dir_ba_x * extra_diff,
                                        pa.y - dir_ba_y * extra_diff,
                                        pa.z - dir_ba_z * extra_diff)
                else:
                    ed = extra_diff * 0.5
                    pa.set_position_vec(pa.x - dir_ba_x * ed,
                                        pa.y - dir_ba_y * ed,
                                        pa.z - dir_ba_z * ed)
                    pb.set_position_vec(pb.x + dir_ba_x * ed,
                                        pb.y + dir_ba_y * ed,
                                        pb.z + dir_ba_z * ed)

    def build_spatial_map(self):
        # map from float : list of point masses
        self.map.clear()
        for pt in self.pts:
            pthash = self.hash_position(pt.x, pt.y, pt.z)
            if pthash not in self.map:
                self.map[pthash] = []
            self.map[pthash].append(pt)

    def hash_position(self, double ptx, double pty, double ptz):
        cdef double w = 3 * self.dx
        cdef double h = 3 * self.dy
        cdef double t = max(w, h)
        return (31*31) * math.floor(ptx / w) + 31 * math.floor(pty / h) + math.floor(ptz / t)

    def self_collide(self, pt, int simulation_steps, double thickness):
        if pt.pinned:
            return
        cdef int pthash    = self.hash_position(pt.x, pt.y, pt.z)
        cdef double thresh = 2.0 * thickness

        if pthash in self.map:
            tot_corr_x = 0.0
            tot_corr_y = 0.0
            tot_corr_z = 0.0
            n = 0
            for candidate in self.map[pthash]:
                if pt is candidate:
                    continue
                distance = fastnorm(pt.x - candidate.x,
                                    pt.y - candidate.y,
                                    pt.z - candidate.z)
                if distance <= thresh:
                    factor = (thresh - distance) / distance
                    tot_corr_x += (pt.x - candidate.x) * factor
                    tot_corr_y += (pt.y - candidate.y) * factor
                    tot_corr_z += (pt.z - candidate.z) * factor
                    n += 1
            if n != 0:
                n = float(n)
                corr_x = tot_corr_x / n / simulation_steps
                corr_y = tot_corr_y / n / simulation_steps
                corr_z = tot_corr_z / n / simulation_steps
                pt.set_position_vec(pt.x + corr_x,
                                    pt.y + corr_y,
                                    pt.z + corr_z)

    def _handle_plane_collision(self, pt, double p_friction, double surface_offset):
        """Handle cloth-plane collisions.

        Normally, we use dot products between a point's current position and the
        plane's normal vector, to detect if we're 'colliding' with it. Here, we
        can look at the z-axis because we assume our planes have normal (0,0,1).

        Use p_friction as friction with the surface. If 1, then points that
        touch the plane surface should be fixed, as we'd be reverting back to
        the older point position.
        """
        if pt.pinned or pt.z >= self.minimum_z:
            return
        cdef double t = (self.minimum_z - pt.pz) * 1.0
        tangent_x = pt.px + t * (-0.0)
        tangent_y = pt.py + t * (-0.0)
        tangent_z = pt.pz + t * (-1.0)
        goal_x = tangent_x + surface_offset * 0.0
        goal_y = tangent_y + surface_offset * 0.0
        goal_z = tangent_z + surface_offset * 1.0
        corr_x = goal_x - pt.px
        corr_y = goal_y - pt.py
        corr_z = goal_z - pt.pz
        pt.set_position_vec(pt.px + corr_x * (1. - p_friction),
                            pt.py + corr_y * (1. - p_friction),
                            pt.pz + corr_z * (1. - p_friction))
        #double t = dot(this->point - pm.last_position, this->normal) / dot(-this->normal, this->normal);
        #Vector3D tangent = pm.last_position + t * (-this->normal);
        #Vector3D goal = tangent + SURFACE_OFFSET * this->normal;
        #Vector3D correction = goal - pm.last_position;
        #pm.position = pm.last_position + correction * (1 - this->friction);

    def init_render(self):
        """Publish cloth state to ZeroMQ socket. C++ renderer subscribes to this.
        """
        self.render = True
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:%d" % self.render_port)
        self.socket = socket

    def stop_render(self):
        address = self.socket.getsockopt(zmq.LAST_ENDPOINT)
        self.socket.unbind(address)

    @property
    def have_tear(self):
        return self.cloth_have_tear

    @property
    def allpts_arr(self):
        return np.array([[p.x, p.y, p.z] for p in self.pts])

    @property
    def noncolorpts_arr(self):
        return np.array([[p.x, p.y, p.z] for p in self.pts if p not in self.color_pts])

    @property
    def colorpts_arr(self):
        return np.array([[p.x, p.y, p.z] for p in self.color_pts])

    @property
    def pinnedpts_arr(self):
        return np.array([[p.x, p.y, p.z] for p in self.pts if p.pinned])


class Spring(object):

    def __init__(self, ptA, ptB, springtype):
        self.ptA = ptA
        self.ptB = ptB
        self.type = springtype
        self.rest_length = fastnorm(ptA.x - ptB.x, ptA.y - ptB.y, ptA.z - ptB.z)
