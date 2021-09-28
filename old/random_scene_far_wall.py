#!/usr/bin/env python3
import argparse
from old.depth_camera import DepthCamera
import dm_control
import dm_control.mujoco as mujoco
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2
import depth_camera
from PIL import Image
import pandas as pd
import sys

import open3d as o3d

from pyntcloud import PyntCloud
import pandas as pd
import argparse
from scipy.spatial.transform import Rotation

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

# General
import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
np.random.seed(1)

img_width = 320//2
img_height = 240//2
NUM_CREATURES = 6
BODY_RADIUS = 0.1
BODY_SIZE = (BODY_RADIUS, BODY_RADIUS/2, BODY_RADIUS / 2)
random_state = np.random.RandomState(42)

parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder', help='Folder to create datset in')
args = parser.parse_args()

dataset_folder = args.dataset_folder

def make_quat(rotpos):
  return np.array([np.roll(Rotation.from_euler('xyz', [0, 0, r], degrees=True).as_quat(), 1) for r in rotpos])

def make_arena():
  arena = mjcf.RootElement()
  tglobal = arena.visual.get_children("global")
  tglobal.set_attributes(offwidth=f"{img_width}")
  tglobal.set_attributes(offheight=f"{img_height}")

  checkered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
                              height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
  grid = arena.asset.add('material', name='grid', texture=checkered,
                        texrepeat=[5, 5], reflectance=.2)
  arena.worldbody.add('geom', name="groundplane", type='plane', size=[2, 2, .1], material=grid)
  arena.worldbody.add('camera', name="carcam", euler=[0, -90, -90], pos=[-3, 0, 0.5])

  for idx, x in enumerate([-2, 2]):
    light = arena.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2], name=f"light{idx}")

  return arena

gen_rand_x = lambda: random_state.uniform(-1, 1)
gen_rand_y = lambda: random_state.uniform(-1, 1)
gen_rand_z = lambda: random_state.uniform(0.15, 0.15)
gen_rand_rot = lambda: random_state.uniform(-180, 180)

def gen_pos_orintation():
  return  gen_rand_x(), gen_rand_y(), gen_rand_z(), gen_rand_rot()


class Leg(object):
  """A 2-DoF leg with position actuators."""
  def __init__(self, length, rgba):
    self.model = mjcf.RootElement()

    # Defaults:
    self.model.default.joint.damping = 2
    self.model.default.joint.type = 'hinge'
    self.model.default.geom.type = 'capsule'
    self.model.default.geom.rgba = rgba  # Continued below...

    # Thigh:
    self.thigh = self.model.worldbody.add('body')
    self.hip = self.thigh.add('joint', axis=[0, 0, 1])
    self.thigh.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[length/4])

    # Hip:
    self.shin = self.thigh.add('body', pos=[length, 0, 0])
    self.knee = self.shin.add('joint', axis=[0, 1, 0])
    self.shin.add('geom', fromto=[0, 0, 0, 0, 0, -length], size=[length/5])

    # Position actuators:
    self.model.actuator.add('position', joint=self.hip, kp=10)
    self.model.actuator.add('position', joint=self.knee, kp=10)

# def make_creature(idx, arena):
#   x, y, z, rot = gen_pos_orintation()
#   rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])
#   # body = arena.worldbody.add('body', name=f"body{idx}")
#   # body.add(
#   #     'geom', name=f'torso{idx}', type='ellipsoid', size=BODY_SIZE, rgba=rgba)
#   print("+========================+")
#   print(arena.to_xml_string())

#   hip_site = arena.worldbody.add('site', pos=(x, y, z), euler=[0, 0, rot])
#   l = Leg(length=BODY_RADIUS, rgba=rgba)
#   hip_site.attach(l.model)

  # spawn_site = arena.worldbody.add('site', 
  #   name=f"creaturesite{idx}",
  #   pos=(x, y, z), 
  #   group=3, 
  #   euler=np.array([0, 0, rot]))
  # print("+========================+")
  # print(arena.to_xml_string())
  # spawn_site.attach(body)



  # # Attach to the arena at the spawn sites, with a free joint.
  # attached = spawn_site.attach(body)
  # attach_joint = attached.add('freejoint', name=f"creaturejoint{idx}")
  # body.actuator.add('position', joint=attach_joint, kp=10)

  # return l.model

def make_creature(idx, arena):
  rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])
  x, y, z, rot = gen_pos_orintation()
  model = mjcf.RootElement()
  model.worldbody.add(
      'geom', name=f'torso{idx}', type='ellipsoid', size=BODY_SIZE, rgba=rgba)
  spawn_site = arena.worldbody.add('site', 
    name=f"sitename{idx}",
    pos=(x,y,z), 
    group=3, 
    euler=np.array([0, 0, rot]))
  # Attach to the arena at the spawn sites, with a free joint.
  attached = spawn_site.attach(model)
  attached.add('freejoint', name=f"joint{idx}")
  return model
    

def make_creatures(arena):
  return [make_creature(i, arena) for i in range(NUM_CREATURES)]
  

arena = make_arena()
creatures = make_creatures(arena)
actuators = [a for creature in creatures for a in creature.find_all('actuator')]
print(arena.to_xml_string())

def make_boxs(idx, arena, camera : depth_camera.DepthCamera):
  # print("Arena:")  
  # help(arena.worldbody)
  # print(arena.worldbody.all_children())
  # print(mjcf.RootElement().worldbody)
  mesh = o3d.geometry.TriangleMesh()

  for creature_site in arena.worldbody.get_children("site"):
    x, y, z = creature_site.pos
    _, _, rot = creature_site.euler

    _, _, rotation, translation = camera.camera.matrices()
    global_pos = np.array([x, y, z, 1])
    depth_pos = rotation @ -translation @ global_pos
    xt, yt, zt, _ = depth_pos
    yt, zt = zt, yt
    zt = -zt
    xt, yt = yt, xt
    # np.concatenate([xs.reshape(1, len(xs)), ys.reshape(1, len(ys)), zs.reshape(1, len(zs)), np.ones((1, len(xs)))])

    # print(creature_site.euler)
    cube = o3d.geometry.TriangleMesh.create_box(*[e * 2 for e in BODY_SIZE])
    cube.rotate(Rotation.from_euler('xyz', [0, 0, rot], degrees=True).as_matrix(), 
                cube.get_center())
    cube.translate((xt, yt, zt), relative=False)
    # cube.compute_vertex_normals()
    mesh += cube
  mesh.compute_vertex_normals()
  o3d.io.write_triangle_mesh(f"{dataset_folder}/{idx:06d}_mesh.ply", mesh)


from pathlib import Path
Path(f"{dataset_folder}/").mkdir(parents=True, exist_ok=True)
# # Instantiate the physics and render.
physics = mjcf.Physics.from_mjcf_model(arena)
physics.reset()
for idx in range(1, 5):
  # physics.bind(actuators).ctrl = np.ones(NUM_CREATURES) * physics.data.time * 5000
  # print(physics.bind(actuators).ctrl)
  # # physics.bind(actuators).ctrl = amp * np.sin(freq * physics.data.time + phase)
  # physics.step()
#   # print(physics.data.qpos.shape)
#   # print(physics.data.qpos)
#   # print(physics.data.xquat.shape)
#   # print(physics.data.xquat)
#   xs = gen_rand_xs()
#   ys = gen_rand_ys()
#   zs = gen_rand_zs()
#   rots = gen_rand_rots()
#   quats = make_quat(rots).T
  # with physics.reset_context():
# print(physics.named.data.qpos)
#     physics.data.qpos[0::7] = xs
#     physics.data.qpos[1::7] = ys
#     physics.data.qpos[2::7] = zs
#     physics.data.qpos[3::7] = quats[0]
#     physics.data.qpos[4::7] = quats[1]
#     physics.data.qpos[5::7] = quats[2]
#     physics.data.qpos[6::7] = quats[3]
#     # physics.data.geom_xpos[1:,1][:] = gen_rand_ys()
#     # physics.data.geom_xpos[1:,2][:] = gen_rand_zs()
#     # physics.data.xquat[1:][:] = make_quad(gen_rand_rots())
    
#   # print(physics.data.xpos)
#   # print(physics.data.xquat)
#   # help(physics.data)

  
  camera = depth_camera.DepthCamera(mujoco.Camera(physics, img_height, img_width, camera_id="carcam"))
  camera.save_ply(f"{dataset_folder}/{idx:06d}.ply")

  make_boxs(idx, arena, camera)

  # boxes = make_object_box(xs, ys, zs, rots, BODY_SIZE, camera)  
  # save_object_boxes(boxes, f"{dataset_folder}/label_2/{idx:06d}.txt")
  # render_object_boxes(boxes,f"{dataset_folder}/{idx:06d}_labels.ply")
  # print("Saved instance {}".format(idx))


