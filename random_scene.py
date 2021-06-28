#!/usr/bin/env python3
import argparse
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

def make_creature(idx):
  rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])
  model = mjcf.RootElement()
  model.worldbody.add(
      'geom', name=f'torso{idx}', type='ellipsoid', size=BODY_SIZE, rgba=rgba)
  return model

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

  return arena

gen_rand_xs = lambda: random_state.uniform(-1, 1, NUM_CREATURES)
gen_rand_ys = lambda: random_state.uniform(-1, 1, NUM_CREATURES)
gen_rand_zs = lambda: random_state.uniform(0.15, 0.15, NUM_CREATURES)
gen_rand_rots = lambda: random_state.uniform(-180, 180, NUM_CREATURES)

def gen_pos_orintations():
  return  gen_rand_xs(), gen_rand_ys(), gen_rand_zs(), gen_rand_rots()

def make_creatures(arena):
  for idx, x in enumerate([-2, 2]):
    light = arena.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2], name=f"light{idx}")

  creatures = [make_creature(i) for i in range(NUM_CREATURES)]
  return creatures


def add_creatures_to_arena(arena, creatures):
  xs, ys, zs, rots = gen_pos_orintations()
  for i, creature in enumerate(creatures):
    spawn_pos = (xs.flat[i], ys.flat[i], zs.flat[i])
    spawn_site = arena.worldbody.add('site', 
    name=f"sitename{i}",
    pos=spawn_pos, 
    group=3, 
    euler=np.array([0, 0, rots.flat[i]]))
    # Attach to the arena at the spawn sites, with a free joint.
    attached = spawn_site.attach(creature)
    attached.add('freejoint', name=f"joint{i}")
    print(creature.geom_pos)

arena = make_arena()
creatures = make_creatures(arena)
add_creatures_to_arena(arena, creatures)

def make_box(camera: depth_camera.DepthCamera):
  mesh = o3d.geometry.TriangleMesh()

  for _ in range(NUM_CREATURES):
    cube = o3d.geometry.TriangleMesh.create_box(*BODY_SIZE)
    cube.translate(
        (
            np.random.uniform(0,1),
            np.random.uniform(0,1),
            np.random.uniform(0,1),
        ),
        relative=False,
    )
    # cube.compute_vertex_normals()
    mesh += cube
  mesh.compute_vertex_normals()
  o3d.io.write_triangle_mesh(f"{dataset_folder}/cube_mesh.ply", mesh)
  # o3d.visualization.draw_geometries([cube])


def make_object_box(xs, ys, zs, rots, body_size, camera: depth_camera.DepthCamera):
  assert(len(xs) == len(ys))
  assert(len(zs) == len(ys))
  assert(len(rots) == len(ys))

  _, _, rotation, translation = camera.camera.matrices()

  global_pos = np.concatenate([xs.reshape(1, len(xs)), ys.reshape(1, len(ys)), zs.reshape(1, len(zs)), np.ones((1, len(xs)))])

  depth_pos = rotation @ -translation @ global_pos

  xs, ys, zs, _ = depth_pos
  ys, zs = zs, ys
  zs = -zs
  xs, ys = ys, xs

  assert(len(xs) == len(ys))
  assert(len(zs) == len(ys))
  assert(len(rots) == len(ys))

  label_lst = []
  # 1 type
  label_lst.append(np.array(["Car"]*len(xs)))
  # 1 truncated
  label_lst.append(np.zeros(len(xs)))
  # 1 occluded
  label_lst.append(np.zeros(len(xs)).astype(np.int32))
  # 1 alpha
  label_lst.append(rots)
  # 4 bbox
  label_lst.append(np.zeros(len(xs)))
  label_lst.append(np.zeros(len(xs)))
  label_lst.append(np.zeros(len(xs)))
  label_lst.append(np.zeros(len(xs)))
  # 3 dimensions
  label_lst.append(np.array([body_size[0]] * len(xs)))
  label_lst.append(np.array([body_size[1]] * len(xs)))
  label_lst.append(np.array([body_size[2]] * len(xs)))
  # 3 location
  label_lst.append(xs)
  label_lst.append(ys)
  label_lst.append(zs)
  # 1 rotation
  label_lst.append(rots)
  # 1 score
  label_lst.append(np.ones(len(xs)))
  arr = np.array(label_lst).T
  assert(arr.shape == (len(xs), 16))
  return arr

def save_object_boxes(arr_lst, filename):
  f = open(filename, 'w')
  for arr in arr_lst:
    f.write(' '.join(arr))
    f.write('\n')
  f.close()

def render_object_boxes(arr_lst, filename):
  filtered_lst = arr_lst[:,11:14].astype(np.float32)
  PyntCloud(pd.DataFrame(data=filtered_lst,
        columns=["x", "y", "z"])).to_file(filename)


from pathlib import Path
Path(f"{dataset_folder}/velodyne").mkdir(parents=True, exist_ok=True)
Path(f"{dataset_folder}/image_2").mkdir(parents=True, exist_ok=True)
Path(f"{dataset_folder}/calib").mkdir(parents=True, exist_ok=True)
Path(f"{dataset_folder}/label_2").mkdir(parents=True, exist_ok=True)
Path(f"{dataset_folder}/ImageSets").mkdir(parents=True, exist_ok=True)

# Instantiate the physics and render.
physics = mjcf.Physics.from_mjcf_model(arena)
for idx in range(1, 2):
  # print(physics.data.qpos.shape)
  # print(physics.data.qpos)
  # print(physics.data.xquat.shape)
  # print(physics.data.xquat)
  xs = gen_rand_xs()
  ys = gen_rand_ys()
  zs = gen_rand_zs()
  rots = gen_rand_rots()
  quats = make_quat(rots).T
  with physics.reset_context():
    physics.data.qpos[0::7] = xs
    physics.data.qpos[1::7] = ys
    physics.data.qpos[2::7] = zs
    physics.data.qpos[3::7] = quats[0]
    physics.data.qpos[4::7] = quats[1]
    physics.data.qpos[5::7] = quats[2]
    physics.data.qpos[6::7] = quats[3]
    # physics.data.geom_xpos[1:,1][:] = gen_rand_ys()
    # physics.data.geom_xpos[1:,2][:] = gen_rand_zs()
    # physics.data.xquat[1:][:] = make_quad(gen_rand_rots())
    
  # print(physics.data.xpos)
  # print(physics.data.xquat)
  # help(physics.data)

  
  camera = depth_camera.DepthCamera(mujoco.Camera(physics, img_height, img_width, camera_id="carcam"))
  camera.save_ply(f"{dataset_folder}/{idx:06d}.ply")
  camera.save_kitti_pc(f"{dataset_folder}/velodyne/{idx:06d}.bin", True)
  camera.save_img(f"{dataset_folder}/image_2/{idx:06d}.png")
  camera.save_calibration(f"{dataset_folder}/calib/{idx:06d}.txt")

  make_box(camera)

  boxes = make_object_box(xs, ys, zs, rots, BODY_SIZE, camera)  
  save_object_boxes(boxes, f"{dataset_folder}/label_2/{idx:06d}.txt")
  render_object_boxes(boxes,f"{dataset_folder}/{idx:06d}_labels.ply")
  print("Saved instance {}".format(idx))


