#!/usr/bin/env python3
import os
# Without this set system tries to use OpenGL and crashes
os.environ["MUJOCO_GL"] = "egl"

import argparse

import copy
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pandas.core import frame
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
import multiprocessing

np.random.seed(1)

img_width = 640
img_height = 480
NUM_OBJECTS = 4
random_state = np.random.RandomState(42)

parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder', help='Folder to create datset in')
parser.add_argument('--num_cpus',
                    help='Number of CPUs to use',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    required=False)
parser.add_argument('--num_rollouts',
                    help='Number of rollouts to perform',
                    type=int,
                    default=1000,
                    required=False)
args = parser.parse_args()

dataset_folder = args.dataset_folder


def make_quat(rotpos):
    return np.array([
        np.roll(
            Rotation.from_euler('xyz', [0, 0, r], degrees=True).as_quat(), 1)
        for r in rotpos
    ])


def make_arena():
    arena = mjcf.RootElement()
    arena.model = "arena"
    arena.option.set_attributes(integrator="RK4")
    tglobal = arena.visual.get_children("global")
    tglobal.set_attributes(offwidth=f"{img_width}")
    tglobal.set_attributes(offheight=f"{img_height}")
    tquality = arena.visual.get_children("quality")
    tquality.set_attributes(offsamples=0)

    checkered = arena.asset.add('texture',
                                type='2d',
                                builtin='checker',
                                width=300,
                                height=300,
                                rgb1=[.2, .3, .4],
                                rgb2=[.3, .4, .5])
    grid = arena.asset.add('material',
                           name='grid',
                           texture=checkered,
                           texrepeat=[5, 5],
                           reflectance=.2)
    arena.worldbody.add('geom',
                        name="groundplane",
                        type='plane',
                        size=[20, 20, .1],
                        material=grid)
    arena.worldbody.add('camera',
                        name="carcam",
                        euler=[0, -90, -90],
                        pos=[-4, 0, 0.5])

    for idx, x in enumerate([-2, 2]):
        light = arena.worldbody.add('light',
                                    pos=[x, -1, 3],
                                    dir=[-x, 1, -2],
                                    name=f"light{idx}")

    return arena


gen_rand_x = lambda: random_state.uniform(-1, 1)
gen_rand_y = lambda: random_state.uniform(-1, 1)
gen_rand_z = lambda: random_state.uniform(0.15, 0.15)
gen_rand_rot = lambda: random_state.uniform(-180, 180)


def gen_pos_orintation():
    return gen_rand_x(), gen_rand_y(), gen_rand_z(), gen_rand_rot()


def make_objects(arena, num_objs, cls=flying_block.FlyingBlock):
    objs = [cls(idx) for idx in range(num_objs)]
    for o in objs:
        x, y, z, _ = gen_pos_orintation()
        o.add_to_arena(arena, (x, y, z))
    return objs


def run_simulation(run_idx, framerate=8, duration=2):
    arena = make_arena()
    objects = make_objects(arena, NUM_OBJECTS)

    physics = mjcf.Physics.from_mjcf_model(arena)
    physics.reset()
    steps_per_sec = 1 // physics.timestep()  # (Hz)
    steps_per_render = steps_per_sec // framerate
    total_steps = steps_per_render * framerate * duration

    frame_idx = framerate * duration * run_idx

    physics = mjcf.Physics.from_mjcf_model(arena)
    physics.reset()
    for o in objects:
        o.initialize_velocities(physics, run_idx)
    camera = depth_camera.DepthCamera(
        mujoco.Camera(physics, img_height, img_width, camera_id="carcam"))

    step = 0
    while step < total_steps:
        step += 1
        physics.step()
        if step % steps_per_render == 0:
            print(f"Frame {frame_idx:06d}")
            camera.render_video_frame()
            camera.save_np_and_classes(
                f"{dataset_folder}/{frame_idx:06d}_pc.pkl",
                f"{dataset_folder}/{frame_idx:06d}_classes.pkl")
            frame_idx += 1

    camera.save_video(
        f"{dataset_folder}/scene_video_rollout_{run_idx:03d}.mp4", framerate)


from pathlib import Path

Path(f"{dataset_folder}/").mkdir(parents=True, exist_ok=True)

with multiprocessing.Pool(args.num_cpus) as p:
    p.map(run_simulation, range(args.num_rollouts))
