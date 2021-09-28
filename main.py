#!/usr/bin/env python3
import argparse


import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import open3d as o3d
import dm_control
import dm_control.mujoco as mujoco
from dm_control import mjcf
import numpy as np
import depth_camera
import spider
import flying_block
import argparse
from scipy.spatial.transform import Rotation


np.random.seed(1)

img_width = 640
img_height = 480
NUM_CREATURES = 4
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
  tquality = arena.visual.get_children("quality")
  tquality.set_attributes(offsamples=0)

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

def make_objects(arena, num_objs, cls=flying_block.FlyingBlock):
  objs = [cls() for _ in range(num_objs)]
  for o in objs:
    x, y, z, _ = gen_pos_orintation()
    o.add_to_arena(arena, (x, y, z))
  return objs

arena = make_arena()
objects = make_objects(arena, NUM_CREATURES, cls=spider.Spider)
actuators = [a for obj in objects for a in obj.actuators()]

def make_cylinders(idx, physics, creatures, camera : depth_camera.DepthCamera):
  mesh = o3d.geometry.TriangleMesh()
  model_names = ['unnamed_model/'] + ['unnamed_model_{}/'.format(idx) for idx in range(1, len(creatures))]
  for model_name, creature in zip(model_names, creatures):
    x, y, z, r1, r2, r3, r4 = physics.named.data.qpos[model_name]
    _, _, rotation, translation = camera.camera.matrices()
    global_pos = np.array([x, y, z, 1])
    depth_pos = rotation @ -translation @ global_pos
    xt, yt, zt, _ = depth_pos
    yt, zt = zt, yt
    zt = -zt
    xt, yt = yt, xt

    cyl = o3d.geometry.TriangleMesh.create_cylinder(creature.body_radius, creature.body_height)
    cyl.rotate(Rotation.from_quat([r2, r3, r4, r1]).as_matrix(), 
                cyl.get_center())
    cyl.translate((xt, yt, zt), relative=False)
    mesh += cyl
  mesh.compute_vertex_normals()
  o3d.io.write_triangle_mesh(f"{dataset_folder}/{idx:06d}_mesh.ply", mesh)


from pathlib import Path
Path(f"{dataset_folder}/").mkdir(parents=True, exist_ok=True)
# print(arena.to_xml_string())
# Instantiate the physics and render.
physics = mjcf.Physics.from_mjcf_model(arena)
physics.reset()
camera = depth_camera.DepthCamera(mujoco.Camera(physics, img_height, img_width, camera_id="carcam"))

duration = 2   # (Seconds)
framerate = 30  # (Hz)
steps_per_sec = 1//physics.timestep() # (Hz)
steps_per_render = steps_per_sec // framerate

print("Timestep:", physics.timestep(), "Framerate:", framerate)
idx = 0
step = 0

freq = 5
phase = 2 * np.pi * random_state.rand(len(actuators))
amp = 2

# for step in range(500):
while physics.data.time < duration:
  step += 1
  if len(actuators) > 0:
    physics.bind(actuators).ctrl = amp * np.sin(freq * physics.data.time + phase)
  physics.step()
  if step % steps_per_render == 0:
    print(f"Frame {idx:06d} of {duration * framerate:06d}")
    camera.render_video_frame()
    camera.save_np_and_classes(f"{dataset_folder}/{idx:06d}_pc.pkl", 
                               f"{dataset_folder}/{idx:06d}_classes.pkl")
    # camera.save_ply(f"{dataset_folder}/{idx:06d}.ply")
    # camera.save_ply(f"{dataset_folder}/{idx:06d}_noback.ply", background_subtract=True)
    # camera.save_img(f"{dataset_folder}/{idx:06d}.png")
    # camera.save_segmentation(f"{dataset_folder}/{idx:06d}_semantics.png")
    # make_cylinders(idx, physics, objects, camera)
    idx += 1

camera.save_video(f"{dataset_folder}/scene_video.mp4", framerate)
