"""Similar to the random policy, but with better obs / action spaces.

Update Summer 2019 (starting in June): using to test rendering.
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
import cv2
import pyrender
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from gym_cloth.envs import ClothEnv
from os.path import join
from collections import defaultdict
np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)
DEG_TO_RAD = np.pi / 180

# ---------------------------------------------------------------------------- #
# IMAGES / RENDERING                                                           #
# ---------------------------------------------------------------------------- #

def _preprocess_depth(dimg):
    """Just like we had earlier for actual bed-making.
    """
    def depth_to_3ch(img, cutoff=1000000):
        w,h = img.shape
        new_img = np.zeros([w,h,3])
        img = img.flatten()
        img[img>cutoff] = 0.0
        img = img.reshape([w,h])
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def depth_scaled_to_255(img):
        assert np.max(img) > 0.0
        img = 255.0/np.max(img)*img
        img = np.array(img,dtype=np.uint8)
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return img

    dimg = depth_to_3ch(dimg)
    dimg = depth_scaled_to_255(dimg)
    # Can optionally process it further.
    #kernel = np.ones((5,5), np.float32) / 25
    #dimg = cv2.filter2D(dimg, -1, kernel)
    #dimg = cv2.blur(dimg, (5,5))
    return dimg


def _save_render_images(scene, rend):
    """Saves images from the render.
    Params: `r`: Offscreen Renderer
    """
    color, depth = rend.render(scene)
    depth_proc = _preprocess_depth(depth)
    num = len([x for x in os.listdir('examples') if x[-4:]=='.png' and 'img_c' in x])
    cname = 'examples/img_c_{}.png'.format(str(num).zfill(3))
    dname = 'examples/img_d_{}.png'.format(str(num).zfill(3))
    cv2.imwrite(cname, color)
    cv2.imwrite(dname, depth_proc)


def _save_trimesh(env, scene, rend):
    """Save the cloth as an obj so we can render it.
    """
    cloth = [[p.x, p.y, p.z] for p in env.cloth.pts]
    cloth = np.array(cloth)
    assert cloth.shape == (625,3), cloth.shape
    wh = 25
    faces = []
    for r in range(wh-1):
        for c in range(wh-1):
            pp = r*wh + c
            faces.append( [pp,   pp+wh, pp+1] )
            faces.append( [pp+1, pp+wh, pp+wh+1] )
    faces = np.array(faces)
    assert faces.shape == (1152,3), faces.shape

    # Save as obj so that we can render in Blender more nicely. :-)
    tm = trimesh.Trimesh(vertices=cloth, faces=faces)
    num = len([x for x in os.listdir('examples') if x[-4:] == '.obj'])
    tm.export('examples/trimesh_cloth_{}.obj'.format(str(num).zfill(3)))

    # Create mesh+node, add node, save images, then remove mesh+node.
    mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
    mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    scene.add_node(mesh_node)
    _save_render_images(scene, rend)
    scene.remove_node(mesh_node)


def _create_scene_and_offscreen_render():
    """We will add (and remove) the meshes later.
    Unfortunately this requires some tuning of the position and rotation.
    """
    scene = pyrender.Scene()
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    # The position / translation [px, py, pz] and then rotation matrix.
    p = [0.45, -0.45, 0.90]
    theta = -45 * DEG_TO_RAD
    RX = np.array([
        [1.0,            0.0,           0.0],
        [0.0,  np.cos(theta), np.sin(theta)],
        [0.0, -np.sin(theta), np.cos(theta)],
    ])
    R = RX
    camera_pose = np.array([
       [R[0,0], R[0,1], R[0,2], p[0]],
       [R[1,0], R[1,1], R[1,2], p[1]],
       [R[2,0], R[2,1], R[2,2], p[2]],
       [   0.0,    0.0,    0.0,  1.0],
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(2), intensity=1.0,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    # Only for debugging
    #v = pyrender.Viewer(scene, use_raymond_lighting=True)
    rend = pyrender.OffscreenRenderer(640, 480)
    return scene, rend


# ---------------------------------------------------------------------------- #
# RUNNING ENVIRONMENT                                                          #
# ---------------------------------------------------------------------------- #

def run(config_file, render_path, file_path, result_path, load_state, max_episodes):
    """Run a policy.

    Note that there are many possible interpretations of 'random' actions.
    It's faster if we have `load_state=True`, so if there isn't a state ready
    to load, then run one call to `env.reset()` to get one.

    Actually, we can also do an analytic one where we grip the highest point.
    That part is trivial, but determining the length and direction can be more
    complicated. We can just use hard-coded rules.
    """
    with open(config_file, 'r') as fh:
        cfg = yaml.load(fh)
        seed = cfg['seed']

    # Save states into local directory, load from nfs diskstation.
    NFS = '/nfs/diskstation/seita/clothsim'
    state_path = join(file_path,"state_init.pkl")
    load_state_path = join(NFS,'state_init_med_49_coverage.pkl')
    random_pol = True
    num_episodes = 0
    stats_all = []

    # Should seed env this way, following gym conventions.
    if load_state:
        env = ClothEnv(config_file, start_state_path=load_state_path)
    else:
        env = ClothEnv(config_file)
    env.seed(seed)
    env.render(filepath=render_path)

    # Fix a Pyrender scene, so that we don't keep re-creating.
    pyr_scene, pyr_rend = _create_scene_and_offscreen_render()

    for ep in range(max_episodes):
        # Do one one episode and put information in `stats_ep`. Save starting state.
        obs = env.reset()
        env.save_state(state_path)
        stats_ep = defaultdict(list)
        done = False
        num_steps = 0

        while not done:
            if random_pol:
                #action = env.get_random_action(atype='over_xy_plane')
                action = env.get_random_action(atype='touch_cloth')
            else:
                raise NotImplementedError()
            _save_trimesh(env, pyr_scene, pyr_rend)
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

    assert len(stats_all) == max_episodes, len(stats_all)
    with open(result_path, 'wb') as fh:
        pickle.dump(stats_all, fh)
    if env.render_proc is not None:
        env.render_proc.terminate()
        env.cloth.stop_render()


if __name__ == "__main__":
    result_pkl   = 'results-{}.pkl'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    # Each time we use environment, we need to pass in some configuration.
    file_path    = os.path.dirname(os.path.realpath(__file__))
    cfg_file     = join(file_path, '../cfg/demo_render.yaml')
    render_path  = join(file_path, '../render/build')
    result_path  = join(file_path, '../logs/{}'.format(result_pkl))

    # Better if `load_state = False` if using analytic policy, otherwise no variation.
    load_state   = False
    max_episodes = 1
    run(cfg_file, render_path, file_path, result_path, load_state, max_episodes)
