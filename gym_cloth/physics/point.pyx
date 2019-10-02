# cython: profile=True
"""
A class that simulates a point mass.
A cloth is made up of a collection of these interacting with each other.
"""
from math import sqrt
import numpy as np

# Can debug if cython working
import cython
if cython.compiled:
    print("Yes, cython compiled.")
else:
    print("Just a lowly interpreted script.")


class Point(object):

    def __init__(self,
                 double x,
                 double y,
                 double z,
                 double boundsx,
                 double boundsy,
                 double boundsz,
                 double identity_0=-1,
                 double identity_1=-1):
        """Initializes an instance of a particle.

        Note: `identity` is used for sending particles to renderer. Also,
        adding 'original' starting coordinate for making analytic policies
        easier to implement.
        """
        self.x  = x
        self.y  = y
        self.z  = z
        self.px = x
        self.py = y
        self.pz = z
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.boundsx = boundsx
        self.boundsy = boundsy
        self.boundsz = boundsz
        self.identity_0 = identity_0
        self.identity_1 = identity_1
        self.pinned = False
        self.orig_x  = x
        self.orig_y  = y
        self.orig_z  = z

    def __str__(self):
        str = "({:.3f}, {:.3f}, {:.3f})".format(self.x, self.y, self.z)
        return str

    def __repr__(self):
        return str(self)

    def add_force(self, double x, double y, double z):
        """Applies a force to itself.

        Technical note: it shouldn't matter if we enforce pinning or not. 184
        did not enforce pinning, it just added to the forces. Later, in verlet
        integration, we skip over points that are pinned, so the forces are not
        applied (and in the next simulation step, forces are cleared).
        """
        #if not self.pinned:
        self.fx = self.fx + x
        self.fy = self.fy + y
        self.fz = self.fz + z

    def set_position_vec(self, double x, double y, double z):
        self.x = x
        self.y = y
        self.z = z

    def set_last_position_vec(self, double px, double py, double pz):
        self.px = px
        self.py = py
        self.pz = pz

    def reset_force_vector(self):
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
