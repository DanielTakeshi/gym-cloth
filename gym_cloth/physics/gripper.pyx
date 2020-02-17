# cython: profile=True
"""Gripper to grab the cloth. Modeled based on the Tensioner code.
"""
import numpy as np
import os, sys


class Gripper(object):

    def __init__(self, cloth, double grip_radius, double height, double thickness):
        """A gripper is designed to act on a particular `cloth` input.

        When gripping, we need to grip based on some target (x,y) value.
        Requires interaction with points from `cloth`, and putting grabbed
        points in `self.grabbed_pts`.
        """
        self.cloth = cloth
        self.grip_radius = grip_radius
        self.grabbed_pts = []
        self.height = height
        self.thickness = thickness
        self.pinned_pts = [] # for bilateral actions

    def grab_top(self, double x, double y, bilateral=False):
        """
        Grab the highest points in the cylinder with center (x,y)
        and radius grip_radius.
        """
        # scan down from z = height in increments of thickness until
        # the set of points with x,y in grip radius and height in
        # [z-thickness,z+thickness] is non-empty
        curZ = self.height
        pts = []
        while (curZ > 0):
            for pt in self.cloth.pts:
                if (pt.x-x)*(pt.x-x) + (pt.y-y)*(pt.y-y) < self.grip_radius and \
                        abs(pt.z-curZ) < 2 * self.thickness:
                    pt.pinned = True
                    pts.append(pt)
            if pts:
                break
            curZ -= self.thickness
        if bilateral:
            self.pinned_pts.extend(pts)
        else:
            self.grabbed_pts.extend(pts)

    def grab(self, double x, double y):
        """Grab points at (x,y).

        More accurately, points within a small radius. We pin these points so
        they won't be affected by gravity. Tuning the radius is critical!
        """
        for pt in self.cloth.pts:
            if (pt.x-x)*(pt.x-x) + (pt.y-y)*(pt.y-y) < self.grip_radius:
                pt.pinned = True
                self.grabbed_pts.append(pt)

    def adjust(self, double x, double y, double z):
        """Adjust the (x,y,z) of grabed points, thus adjusting the cloth.

        Here, these are directly added to the positions.
        """
        for pt in self.grabbed_pts:
            pt.px = pt.x
            pt.py = pt.y
            pt.pz = pt.z
            pt.x = x + pt.x
            pt.y = y + pt.y
            pt.z = z + pt.z

    def release(self, bilateral=False):
        """Release gripper, clearing its `grabbed_pts` list.
        """
        if bilateral:
            for pt in self.pinned_pts:
                pt.pinned = False
            self.pinned_pts = []
        else:
            for pt in self.grabbed_pts:
                pt.pinned = False
            self.grabbed_pts = []
