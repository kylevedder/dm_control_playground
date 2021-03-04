#!/usr/bin/env python

import dm_control
import dm_control.mujoco as mujoco
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2
from pyntcloud import PyntCloud
import pandas as pd
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("plots", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate plot mode.")
args = parser.parse_args()


swinging_body = """
<mujoco>
<asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
            width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>  

        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 0">  
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" pos="3 0 2" quat="1 0 0 0" rgba="1 0 0 1"/>
      <geom name="blue_box" type="box" size=".2 .2 .2" pos="0 0 0" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
      <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
    </body>
    <camera pos="3 0 5" euler="30 0 0" fovy="90" name="fixedcam"></camera>
  </worldbody>
</mujoco>
"""

physics = mujoco.Physics.from_xml_path("./model/scene_cam.xml")

object_names = [physics.model.id2name(i, 'geom') for i in range(physics.model.ngeom)]

for o in object_names:
  if len(o) == 0:
    continue
  print("OBJECT NAME", o, flush=True)
  box_pos = physics.named.data.geom_xpos[o]
  print("box_pos")
  print(box_pos)
  box_mat = physics.named.data.geom_xmat[o].reshape(3, 3)
  print("box_mat")
  print(box_mat)
  box_size = physics.named.model.geom_size[o]
  print("box_size")
  print(box_size)
  offsets = np.array([-1, 1]) * box_size[:, None]
  xyz_local = np.stack(itertools.product(*offsets)).T
  xyz_global = box_pos[:, None] + box_mat @ xyz_local




# # Get the world coordinates of the box corners
# box_pos = physics.named.data.geom_xpos['red_box']
# box_mat = physics.named.data.geom_xmat['red_box'].reshape(3, 3)
# box_size = physics.named.model.geom_size['red_box']
# offsets = np.array([-1, 1]) * box_size[:, None]
# xyz_local = np.stack(itertools.product(*offsets)).T
# xyz_global = box_pos[:, None] + box_mat @ xyz_local

# # Camera matrices multiply homogenous [x, y, z, 1] vectors.
# corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
# corners_homogeneous[:3, :] = xyz_global

# Get the camera matrix.
camera = mujoco.Camera(physics, 200, 150, "fixedcam")
depth_img = camera.render(depth=True)

depth_img[depth_img == np.max(depth_img)] = np.NaN

if (args.plots):
  plt.imshow(depth_img)
  plt.colorbar()
  plt.show()

image, focal, rotation, translation = camera.matrices()


pc = cv2.rgbd.depthTo3d(depth_img, image @ focal[:, :3])
pc = pc.reshape(pc.shape[0] * pc.shape[1], 3)
pc = pc[np.isfinite(pc).any(axis=1)]

homgenious_pc = np.concatenate((pc, np.ones((pc.shape[0], 1), dtype=np.float32)), axis=1)

global_pc = homgenious_pc
global_pc = ( -translation @ np.linalg.inv(rotation) @ homgenious_pc.T).T
pc = global_pc

if (args.plots):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  xs, ys, zs = pc[:, 0], pc[:, 1], pc[:, 2]
  ax.set_box_aspect((1,1,1))
  ax.scatter(xs, ys, zs)
  ax.set_xlabel('$X$')
  ax.set_ylabel('$Y$')
  ax.set_zlabel('$Z$')
  plt.show()


PyntCloud(pd.DataFrame(
data=homgenious_pc[:, :3],
columns=["x", "y", "z"])).to_file("raw_pc.ply")

PyntCloud(pd.DataFrame(
data=global_pc[:, :3],
columns=["x", "y", "z"])).to_file("global_pc.ply")

def save_kitti_pc(data, filename):
  assert(data.shape[1] == 4)
  print(data.shape)
  data[:,3] = 1
  data = data.astype(np.float32)
  f = open(filename, "wb")
  for e in data:
    e = e.ravel()
    f.write(e)
  f.close()


save_kitti_pc(global_pc, "pc.bin")

exit(0)


# # print(camera_matrix)
# proj_matrix = np.concatenate((camera_matrix, np.array([[0, 0, 0, 1]])), 0)
# # print(proj_matrix)

# print("vvvvvvvv")
# init_point = corners_homogeneous[:,5:6]
# print(init_point)
# proj_res = proj_matrix @ corners_homogeneous[:,5:6]
# proj_u, proj_v, ones, zrecip = proj_res
# print(proj_u, proj_v, ones, zrecip)
# norm_u, norm_v, norm_ones, norm_zrecip = proj_u / ones, proj_v / ones, ones / ones, zrecip/ ones
# print(norm_u, norm_v, norm_ones, norm_zrecip)
# depth = depth_img[int(norm_u), int(norm_v)]
# print(depth)
# scaled_u, scaled_v = norm_u , norm_v 
# recov_input = np.array([scaled_u, scaled_v, [1], [1.0/depth]])
# print("recov_input:", recov_input)
# init_recovery = depth * (np.linalg.inv(proj_matrix) @ recov_input)
# print(init_recovery)
# x, y, z, one = init_recovery
# print(x/ one, y / one, z / one, one / one)
# exit(0)


# inv_camera_matrix = np.linalg.inv(camera_matrix)

# Project world coordinates into pixel space. See:
# https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
u, v, z = camera_matrix @ corners_homogeneous
# x and y are in the pixel coordinate system.
x = u / z
y = v / z

depth_img = camera.render(depth=True)

print("========")
print(corners_homogeneous[:,5:6])
print(u[5:6], v[5:6], z[5:6])
xint, yint = int(x[5]), int(y[5])
dval = depth_img[xint, yint]
print(x[5:6], y[5:6], dval)
vec = np.array([[xint, yint, 1, 1.0/dval]])
x, y, z, ones = np.linalg.inv(proj_matrix) @ vec.T
print(x, y, z, ones)
print(x/z/ones, y/z/ones, z/z/ones, ones/ones)
print("========")

# print("xs:", xs)
# print("ys:", ys)
# print("s:",s)
# print("x:", x)
# print("y:", y)

# px = x.astype(np.int32)
# py = y.astype(np.int32)
# print("px:", px)
# print("py:", py)


# Render the camera view and overlay the projected corner coordinates.

# print("Depth:", [depth_img[ix, iy] for ix, iy in zip(px, py)])

# plt.imshow(depth_img)
# plt.show()

# proj_matrix = np.concatenate((camera_matrix, np.array([[0, 0, 0, 1]])), 0)
# print(proj_matrix)

# print("Corners ",corners_homogeneous)

# xs, ys, s, ones = proj_matrix @ corners_homogeneous

# print("xs:", xs)
# print("ys:", ys)
# print("s:",s)
# print("ones:", ones)

plt.imshow(depth_img)
print("X:", x)
print("Y:",y)
print(x[5:6], y[5:6])
plt.plot(x[5:6], y[5:6], '+', c='w')
# plt.gca().set_axis_off()
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(point_vector[:, 0], point_vector[:, 1], point_vector[:, 2])
# plt.show()


# xs_tile = np.tile(np.array(range(depth_img.shape[0])), (depth_img.shape[1], 1)).T
# print(xs_tile)
# print(xs_tile.shape)

# ys_tile = np.tile(np.array(range(depth_img.shape[1])), (depth_img.shape[0], 1))
# print(ys_tile)
# print(ys_tile.shape)

# ex_xs_tile = np.expand_dims(xs_tile, 0)
# ex_ys_tile = np.expand_dims(ys_tile, 0)

# idxs_tile = np.concatenate((ex_xs_tile, ex_ys_tile), 0)
# scaled_idxs_tile = np.copy(idxs_tile)

# scaled_idxs_tile[0, :, :] = scaled_idxs_tile[0, :, :] * depth_img
# scaled_idxs_tile[1, :, :] = scaled_idxs_tile[1, :, :] * depth_img



# xys_camera_frame = np.concatenate((scaled_idxs_tile, np.expand_dims(depth_img, 0)), 0)

# point_vector = xys_camera_frame

# print(xys_camera_frame.shape)

# print(xys_camera_frame.reshape((3, np.prod(xys_camera_frame.shape[1:]))).shape)

# print(camera_matrix.shape)

# point_vector = camera_matrix.T @ xys_camera_frame.reshape((3, np.prod(xys_camera_frame.shape[1:]))) 
# print(point_vector)

# print(.shape)

# res = np.concatenate((xs_tile, ys_tile), 2)
# print(res)

# print(pixels.shape)

# pixel_vector = pixels.reshape(pixels.shape[0] * pixels.shape[1], 3)

# pxs = pixel_vector[:, 0]
# pys = pixel_vector[:, 1]
# pdepth = pixel_vector[:, 2]


# print(camera_matrix.shape, pixel_vector.shape)

# point_vector = pixel_vector @ camera_matrix



# ax.imshow(pixels)
# ax.plot(x, y, '+', c='w')
# ax.set_axis_off()
# plt.show()

# camera = mujoco.Camera(physics, 200, 200)
# pixels = camera.render(depth=True)
# print(camera.matrix)
# # pixels[pixels >= physics.model.stat.extent * physics.model.vis.map_.zfar ] = np.NaN
# print(physics.model.stat.extent * physics.model.vis.map_.zfar )
# plt.imshow(pixels)
# plt.colorbar()
# plt.show()

