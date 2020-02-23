#!/usr/bin/env blender --python
"""
Note how to properly pass an argument to blender:
https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
Also refer to the 2.79 documentation: https://docs.blender.org/api/2.79/

This does not use the `np_random` object we set, so the same random seed means
this might not be reproducible across different trials. However, I think that
is fine?
"""
import bpy
import pickle
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import os
import argparse
import sys
import numpy as np
from numpy.random import normal as rn
from numpy.random import uniform as unif
RAD_TO_DEG = 180. / np.pi
DEG_TO_RAD = np.pi / 180.

# ------------------------------------------------------------------------------
# Domain randomization parameters. Values for numpy random noise need to be
# above 0, so if we're not doing any randomization, just set it to be 1e-5.
# Some reasonable defaults are shown here. Outside of this code, the only other
# domain randomization might be if we want to add noise to the loaded images,
# and it may be easier to do in gym_cloth's environment code.
# ------------------------------------------------------------------------------
# NOTE / TODO: we may modify this later to support saving an original and then
# a noisy version. The OpenAI Dactyl paper said they only applied observation
# noise to the policy input, and not to the value network. This makes sense
# since the value network in DDPG is not used in deployment.
# ------------------------------------------------------------------------------
# To disable domain randomization, set ADD_DOM_RAND = False. Note that I also
# have a variable inside cloth_env.py which does cv2 image processing, because
# I can't import it here due to a different python package.
# ------------------------------------------------------------------------------
ADD_DOM_RAND = False # IF ENABLING OR DISABLING, CHECK cloth_env.__init__ !!
EPS = 1e-5
DR = {
    '_CAMERA_POS_X': EPS,
    '_CAMERA_POS_Y': EPS,
    '_CAMERA_POS_Z': EPS,
    '_CAMERA_DEG_X': EPS,
    '_CAMERA_DEG_Y': EPS,
    '_CAMERA_DEG_Z': EPS,
    '_SPECULAR_MAX': EPS,
}


def set_dom_rand(use_dom_rand):
    global ADD_DOM_RAND
    global DR
    if use_dom_rand.lower() == 'true':
        ADD_DOM_RAND = True
        DR['_CAMERA_POS_X'] = 0.04  #
        DR['_CAMERA_POS_Y'] = 0.04  #
        DR['_CAMERA_POS_Z'] = 0.04  #
        DR['_CAMERA_DEG_X'] = 0.90  # consider using 0.80?
        DR['_CAMERA_DEG_Y'] = 0.90  # consider using 0.80?
        DR['_CAMERA_DEG_Z'] = 0.90  # consider using 0.80?
        DR['_SPECULAR_MAX'] = 0.00  # consuder using 0.10?


def set_active(obj):
    #This only works for 2.8
    bpy.context.view_layer.objects.active = obj


def load_mesh(name):
    """Load mesh from our obj, and return the mesh name for future usage.

    Parameters
    ----------
    name: str
        Must be the full absolute path to the .obj file. We take its base name
        and strip the '.obj' at the end to get the mesh name that blender sees.
    """
    if name[-4:] == ".obj":
        bpy.ops.import_scene.obj(filepath=name)
        mesh_name = (os.path.basename(name)).replace('.obj','')
        return mesh_name
    else:
        raise ValueError("{} not an obj file".format(name))


def set_camera_pose(dim_height, dim_width, camera_pos, camera_deg):
    """Set up the camera pose, potentially applying domain randomization.

    Defaults for the top-down view are 0.5, 0.5, and 1.5 for x, y, and z,
    respectively.  Defaults for rotations are 0 (radians) for all three angles.
    I think yaw, pitch, and roll but the ordering can be determined by testing
    values, e.g.: https://github.com/BerkeleyAutomation/gym-cloth/issues/28.
    """
    # Ryan: fixed DR
    if ADD_DOM_RAND:
        cp = [float(i) for i in camera_pos.split(",")]
        cd = [float(i) for i in camera_deg.split(",")]
    else:
        cp = [rn(0.,scale=EPS),rn(0.,scale=EPS),rn(0.,scale=EPS)]
        cd = [rn(0.,scale=EPS),rn(0.,scale=EPS),rn(0.,scale=EPS)]

    # Select the camera and make it the active object so that we can manipulate it
    bpy.data.objects['Camera'].select = True
    bpy.context.scene.objects.active = bpy.data.objects['Camera']

    # https://blender.stackexchange.com/questions/86233/blender-resizing-my-image-in-half
    bpy.data.scenes['Scene'].render.resolution_percentage = 100.0
    bpy.context.scene.render.resolution_x = dim_width
    bpy.context.scene.render.resolution_y = dim_height

    # Set the x, y and z location (Top-down view). Daniel: height was 1.5 but let's do 1.45.
    bpy.context.object.location[0] = 0.5  + cp[0]
    bpy.context.object.location[1] = 0.5  + cp[1]
    bpy.context.object.location[2] = 1.45 + cp[2]

    # Set the x, y and z rotation (Top-down view).
    bpy.context.object.rotation_euler[0] = DEG_TO_RAD * (0 + cd[0])
    bpy.context.object.rotation_euler[1] = DEG_TO_RAD * (0 + cd[1])
    bpy.context.object.rotation_euler[2] = DEG_TO_RAD * (0 + cd[2])


def set_floor_pose(floor_mesh_name):
    bpy.data.objects[floor_mesh_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[floor_mesh_name]

    #Set the x, y and z location
    bpy.context.object.location[0] = 0.5
    bpy.context.object.location[1] = 1.25
    # Adi: This value needs to be tweaked because it can change the variance in depth
    # Daniel: Let's tweak this, the height of the background (or 'floor').
    bpy.context.object.location[2] = -0.75  # Adi started this at -0.75.

    #Set the x, y and z rotation
    bpy.context.object.rotation_euler[0] = 0
    bpy.context.object.rotation_euler[1] = 0
    bpy.context.object.rotation_euler[2] = 0


def set_bed_pose(bed_mesh_name):
    bpy.data.objects[bed_mesh_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[bed_mesh_name]

    #Set the x, y and z location
    bpy.context.object.location[0] = 0.5
    bpy.context.object.location[1] = 1.0
    # Adi: This value needs to be tweaked because it can change the variance in depth
    # Daniel and Adi later: let's not tweak this value because it's about the right offset.
    bpy.context.object.location[2] = -0.55

    #Set the x, y and z rotation
    bpy.context.object.rotation_euler[0] = 0
    bpy.context.object.rotation_euler[1] = 0
    bpy.context.object.rotation_euler[2] = 0


def set_bed_color(bed_mesh_name, color):
    """Set background plane color from the given mesh.
    """
    bpy.data.objects[bed_mesh_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[bed_mesh_name]

    bpy.data.materials.new(name="bed_material")
    bpy.context.active_object.data.materials.append(bpy.data.materials['bed_material'])
    bpy.context.object.active_material.use_nodes = False

    # Daniel: earlier we had this:
    c = np.array([255.0, 255.0, 255.0])
    # Daniel: https://github.com/BerkeleyAutomation/gym-cloth/issues/37
    c = np.array([1.0, 1.0, 1.0])

    if ADD_DOM_RAND:
        # Daniel: but I feel like this is slight better. Using [0,0,0] means it's
        # not pitch black due to some light, I think. But 0.5 seems reasonable.
        # Hmm ... doing a uniform one seems to heavily emphasize blue-ish and
        # pink-ish colors? Is it better to keep at 0.5 and switch the lighting?
        #c = np.array([0.5, 0.5, 0.5])
        # Eh we can try this:
        # c = np.random.uniform(low=0.4, high=0.6, size=(3,))
        # Ryan: constant DR
        c = np.array([float(i) for i in color.split(",")])

    bpy.context.object.active_material.diffuse_color = (c[0], c[1], c[2])


def set_cloth_color(mesh_name, color_noise, specular_max, init_type, init_side, backfacing=False):
    """Set the cloth color.

    Can set the 'specular intensity' to adjust the brightness of the
    reflection. Values are in [0,1], and we probably want smaller values.

    Parameters
    ----------
    init_type: str
        Tier of initialization we use for cloth.
    init_side: int
        See physics/cloth.pyx, we randomize this for dropping cloth on one of
        two sides of the plane, but it only has an effect if init_type is a
        certain tier (currently tier2). If init_side=1 then we drop on the left
        side, else (-1) we drop on the right side, and it's only for the left
        side when we have to adjust the colors.
    backfacing: bool
        We may (or may not) want to color the front and back in the same way,
        set `backfacing=True` to make the two sides have different colors. We
        need to correct for colors if using starting states that involve cloth
        dropping from different sides of the plane.
    """
    #Select the cloth and make it the active object
    bpy.data.objects[mesh_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[mesh_name]

    bpy.data.materials.new(name="cloth_material")
    bpy.context.active_object.data.materials.append(bpy.data.materials['cloth_material'])
    bpy.data.materials["cloth_material"].specular_intensity = float(specular_max)

    if backfacing:
        bpy.context.object.active_material.use_nodes = True
        node_tree = bpy.context.object.active_material.node_tree
        node_tree.nodes.remove(node_tree.nodes['Material'])
        node_tree.nodes.remove(node_tree.nodes['Output'])
        geometry_node = node_tree.nodes.new("ShaderNodeGeometry")
        mix_node = node_tree.nodes.new("ShaderNodeMixRGB")
        material_node = node_tree.nodes.new("ShaderNodeMaterial")
        material_node.material = bpy.data.materials["cloth_material"]
        output_node = node_tree.nodes.new("ShaderNodeOutput")

        links = node_tree.links
        link_one = links.new(geometry_node.outputs[8], mix_node.inputs[0])
        link_two = links.new(mix_node.outputs[0], material_node.inputs[0])
        link_three = links.new(material_node.outputs[0], output_node.inputs[0])

        # Color the two cloth sides, making adjustments if using tier2 as needed.
        front = 1
        back = 2
        if init_type == 'tier2' and init_side == -1:
            front = 2
            back = 1

        # Daniel: older, green-ish color.
        #b = [0.474, 0.500, 0.075, 1]

        # We may want the same noise applied on the front and back.
        # Ryan: get fixed noise from parameter
        n1 = np.array([float(i) for i in color_noise.split(",")])
        #n1 = np.random.uniform(low=-0.35, high=0.35, size=(3,))
        if ADD_DOM_RAND:
            b = np.array([0.070, 0.300, 0.900]) + n1
            f = np.array([0.070, 0.050, 0.600]) + n1
        else:
            b = np.array([0.070, 0.300, 0.900])
            f = np.array([0.070, 0.050, 0.600])
        b = np.minimum( np.maximum(b, 0.0), 1.0 )
        f = np.minimum( np.maximum(f, 0.0), 1.0 )
        node_tree.nodes["Mix"].inputs[back].default_value  = (b[0], b[1], b[2], 1)
        node_tree.nodes["Mix"].inputs[front].default_value = (f[0], f[1], f[2], 1)
    else:
        bpy.context.object.active_material.use_nodes = False
        bpy.context.object.active_material.diffuse_color = (0.8, 0.03, 0.05)


def set_cloth_texture(mesh_name):
    pass


def set_camera_focal_length():
    #Select the camera and make it the active object
    bpy.data.objects['Camera'].select = True
    bpy.context.scene.objects.active = bpy.data.objects['Camera']

    #Set the focal length
    bpy.context.object.data.lens = 40

    #Set the sensor width to 36mm
    bpy.context.object.data.sensor_width = 36


def set_camera_optical_center():
    #Select the camera and make it the active object
    bpy.data.objects['Camera'].select = True
    bpy.context.scene.objects.active = bpy.data.objects['Camera']

    #Set the optical center (x,y) (I think this is the optical center but I'm not actually sure)
    bpy.context.object.data.shift_x = 0.0
    bpy.context.object.data.shift_y = 0.0


def set_lighting():
    pass


def render_image(obj_path, occlusion_vec):
    """Render the image.

    It might be easier to apply domain randomization in gym_cloth if we want to
    add Gaussian noise to the data. Here, the render result has type
        <class 'bpy.types.Image'>
    and in gym_cloth's environment code, we can directly load a numpy array.

    AH, also note that Blender python doesn't have cv2 ... yeah we have to do
    it in gym-cloth.
    """
    img_path = obj_path.replace('.obj','.png')
    #Adi: Saving the occlusion state in a pickle file as well
    occlusion_path = obj_path.replace('.obj', '')
    with open(occlusion_path, 'wb') as fp:
        pickle.dump(occlusion_vec, fp, protocol=2)

    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(filepath=img_path)


def BVHTreeAndVerticesInWorldFromObj(obj):
    mWorld = obj.matrix_world
    vertsInWorld = [mWorld * v.co for v in obj.data.vertices]
    bvh = BVHTree.FromPolygons(vertsInWorld, [p.vertices for p in obj.data.polygons])
    return bvh, vertsInWorld


def get_occlusion_vec(mesh_name):
    #Checking which corners are occluded in the current camera view (Using my custom 'ray_cast' method because I can't get 'limit selection to visible' to work via Python)
    bpy.data.objects[mesh_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[mesh_name]
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            bpy.ops.object.mode_set(mode='EDIT')
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            area.spaces[0].use_occlude_geometry = False #I actually don't think we need this anymore because we're finding visible vertices in a different way now
            bpy.ops.mesh.select_all(action='TOGGLE')

    bpy.ops.object.mode_set(mode='OBJECT')
    cam = bpy.data.objects['Camera']
    obj = bpy.data.objects[mesh_name]

    #Threshold to test if ray cast corresponds to original vertex
    limit = 0.00001

    #Get bvh tree and vertices in world coordinates (was forgetting to tranform into world frame before... that solved it)
    bvh, vertices = BVHTreeAndVerticesInWorldFromObj(obj)
    #Adi: It seems like Blender maps the ll, lr, and ur corners to the same indices but the ul corner is mapped to 48 instead of 24 for some reason.  Need to investigate more.
    #corners = [vertices[0], vertices[24], vertices[600], vertices[624]]
    corners = [vertices[0], vertices[48], vertices[600], vertices[624]]


    origins = []
    for i in [x * 0.01 for x in range(0, 100)]:
        for j in [x * 0.01 for x in range(0, 100)]:
            origin_lyst = [i, j, 160] #The larger the z coord, the better the accuracy based on preliminary tests.  Seems to plateau at around 160 though.
            origin = Vector(origin_lyst)
            origins.append(origin)

    #Boolean vector for ll, ul, lr, ur occlusion state
    occlusion_vec = [True, True, True, True]

    #Adi: Check if any of the 9 vertices in the region of the corner are visible
    ll = [vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[50], vertices[51], vertices[52]]
    ul = [vertices[44], vertices[45], vertices[46], vertices[47], vertices[48], vertices[49], vertices[72], vertices[73], vertices[74]]
    lr = [vertices[550], vertices[551], vertices[552], vertices[575], vertices[576], vertices[577], vertices[600], vertices[601], vertices[602]]
    ur = [vertices[572], vertices[573], vertices[574], vertices[597], vertices[598], vertices[599], vertices[622], vertices[623], vertices[624]]

    #Adi: Check only if the individual corner vertex is visible
    #ll = [vertices[0]]
    #ul = [vertices[48]]
    #lr = [vertices[600]]
    #ur = [vertices[624]]

    #Adi: Check if any of the
    combined = [ur, lr, ll, ul]

    occlusion_thresh = 0.4 #Should be set to 0 if only one of the vertices in the area needs to be visible in order label the corner as visible
    for i, area in enumerate(combined):
        nb_hits = 0
        for origin in origins:
            for v in area:
                location, normal, index, distance = bvh.ray_cast(origin, (v - origin).normalized())
                #If the ray hits something and if this hit is within the threshold of the desired vertex
                if location and (v - location).length < limit:
                    nb_hits += 1
                    #occlusion_vec[i] = False
                    #break
                if (nb_hits/len(ll)) > occlusion_thresh:
                    occlusion_vec[i] = False
                    break

    del bvh
    return occlusion_vec


def compute_depth():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layer_node = tree.nodes.new("CompositorNodeRLayers")
    normalize_node = tree.nodes.new("CompositorNodeNormalize")
    compositor_node = tree.nodes.new("CompositorNodeComposite")

    links = tree.links
    link_one = links.new(render_layer_node.outputs[2], normalize_node.inputs[0])
    link_two = links.new(normalize_node.outputs[0], compositor_node.inputs[0])


def main():
    # Get the .obj file as a command line argument, after a double dash.
    argv = sys.argv
    argv = argv[argv.index('--')+1:]
    assert len(argv) >= 1, argv
    obj_path = argv[0]
    dim_height = int(argv[1])
    dim_width = int(argv[2])
    init_side = int(argv[3])
    init_type = argv[4]
    bed_obj_path = argv[5]
    #Adi: Adding new flag for corner revealing demonstrator (default is False)
    oracle_reveal = argv[6]
    #Adi: Adding new flag for training on depth images
    use_depth = argv[7]
    #Adi: Adding new path for floor obj to get more realistic depth images
    floor_obj_path = argv[8]
    use_dom_rand = argv[9]

    # Ryan: Adding new flags for fixed DR params from cloth_env.py
    color = argv[10]
    color_noise = argv[11]
    camera_pos = argv[12]
    camera_deg = argv[13]
    specular_max = argv[14]

    #Set domain randomization parameters if using domain randomization
    set_dom_rand(use_dom_rand)

    # Delete the starting cube
    bpy.ops.object.delete(use_global=False)

    # Load cloth mesh and get its 'mesh name' from the 'bpy scene'.
    mesh_name = load_mesh(obj_path)

    #Select the mesh and make it the active object so that we can manipulate it
    bpy.data.objects[mesh_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[mesh_name]

    #Set cloth x rotation to 0 (this is so that the cloth mesh is in the starting pose we want)
    bpy.context.object.rotation_euler[0] = 0

    #Make the cloth mesh smooth
    bpy.ops.object.shade_smooth()

    # Hard-coded path to `blender/frame0.obj` in the gym-cloth repo.
    bed_mesh_name = load_mesh(bed_obj_path)
    set_bed_color(bed_mesh_name, color)
    set_bed_pose(bed_mesh_name)

    #If using depth images, load floor mesh from `blender/floor.obj` in the gym-cloth repo.
    if use_depth == 'True':
        floor_mesh_name = load_mesh(floor_obj_path)
        set_floor_pose(floor_mesh_name)
        #Center the floor mesh
        bpy.data.objects[floor_mesh_name].select = True
        bpy.context.scene.objects.active = bpy.data.objects[floor_mesh_name]
        bpy.data.objects[floor_mesh_name].scale = (2.0, 1.5, 1.0)

    #------Now let's set the scene------
    #Set lighting and shadows (this should fix the black diamonds issue)
    bpy.data.lamps["Lamp"].shadow_method = 'NOSHADOW'
    bpy.data.lamps["Lamp"].falloff_type = 'CONSTANT'
    bpy.data.lamps["Lamp"].energy = 1.5

    #Set the camera pose
    set_camera_pose(dim_height, dim_width, camera_pos=camera_pos, camera_deg=camera_deg)

    #Set the cloth_color
    set_cloth_color(mesh_name, color_noise=color_noise, specular_max=specular_max,
                    init_side=init_side, init_type=init_type, backfacing=True)

    #Set the cloth texture (will do this after I finish the other ones)
    #set_cloth_texture()

    #Set the camera focal length
    set_camera_focal_length()

    #Set the camera optical center
    set_camera_optical_center()

    #Set the lighting
    #set_lighting()

    occlusion_vec = [False, False, False, False]
    if oracle_reveal == 'True':
        occlusion_vec = get_occlusion_vec(mesh_name)

    if use_depth == 'True':
        compute_depth()

    #Render the image
    render_image(obj_path, occlusion_vec)


if __name__ == "__main__":
    main()
