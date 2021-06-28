import cv2
import numpy as np
from PIL import Image
from pyntcloud import PyntCloud
import pandas as pd

class DepthCamera:
  def __init__(self, camera):
    self.camera = camera


  def render_depth(self):
    return self.camera.render(depth=True)

  
  def render_img(self):
    return self.camera.render()


  def render_camera_frame_pc(self):
    depth_img = self.render_depth()
    image, focal, _, _ = self.camera.matrices()
    pc = cv2.rgbd.depthTo3d(depth_img, image @ focal[:, :3])
    pc = pc.reshape(pc.shape[0] * pc.shape[1], 3)
    pc[:, [1, 2]] = pc[:, [2, 1]]
    pc[:, 2] = -pc[:, 2]
    pc = pc[pc[:,1] < 99.5]
    pc[:, [0, 1]] = pc[:, [1, 0]]
    pc = pc[np.isfinite(pc).any(axis=1)]
    return pc


  def render_camera_frame_homogenious_pc(self):
    pc = self.render_camera_frame_pc()
    pc = np.concatenate((pc, np.ones((pc.shape[0], 1), dtype=np.float32)), axis=1)
    return pc


  def render_world_frame_homogenious_pc(self):
    pc = self.render_camera_frame_pc()
    _, _, rotation, translation = self.camera.matrices()
    homgenious_pc = np.concatenate((pc, np.ones((pc.shape[0], 1), dtype=np.float32)), axis=1)
    return ( -translation @ np.linalg.inv(rotation) @ homgenious_pc.T).T


  def render_world_frame_pc(self):
    return self.render_world_frame_homogenious_pc()[:, :3]


  def save_kitti_pc(self, filename, print_bounds=False):
    data = self.render_camera_frame_homogenious_pc()
    if print_bounds:
        pxs, pys, pzs, _ = zip(*data)
        print("Min pxs:", min(pxs), "Max pxs:", max(pxs))
        print("Min pys:", min(pys), "Max pys:", max(pys))
        print("Min pzs:", min(pzs), "Max pzs:", max(pzs))

    assert(data.shape[1] == 4)
    data[:,3] = 1
    data = data.astype(np.float32)
    f = open(filename, "wb")
    for e in data:
      f.write(e.ravel())
    f.close()


  def save_ply(self, filename):
    PyntCloud(pd.DataFrame(data=self.render_camera_frame_pc(),
        columns=["x", "y", "z"])).to_file(filename)


  def save_img(self, filename):
    img = self.render_img()
    im = Image.fromarray(img)
    im.save(filename)


  def save_calibration(self, filename):
    def identity_proj():
      arr = np.zeros((3,4))
      arr[0,0] = 1
      arr[1,1] = 1
      arr[2,2] = 1
      return arr

    def write_arr(f, name, arr):
      f.write(name + ": ")
      for r in arr:    
        f.write(' '.join([str(e) for e in r]) + ' ')
      f.write('\n')

    _, _, rotation, translation = self.camera.matrices()

    f = open(filename, 'w')
    write_arr(f, "P0", self.camera.matrix)
    write_arr(f, "P1", self.camera.matrix)
    write_arr(f, "P2", self.camera.matrix)
    write_arr(f, "P3", self.camera.matrix)
    write_arr(f, "R0_rect", np.eye(3))
    write_arr(f, "Tr_velo_to_cam", -translation @ np.linalg.inv(rotation))
    write_arr(f, "Tr_imu_to_velo", identity_proj())
    f.close()


