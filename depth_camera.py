import cv2
import numpy as np

class DepthCamera:
  def __init__(self, camera):
    self.camera = camera


  def render_depth(self):
    return self.camera.render(depth=True)


  def render_camera_frame_pc(self):
    depth_img = self.render_depth()
    image, focal, _, _ = self.camera.matrices()
    pc = cv2.rgbd.depthTo3d(depth_img, image @ focal[:, :3])
    pc = pc.reshape(pc.shape[0] * pc.shape[1], 3)
    pc = pc[np.isfinite(pc).any(axis=1)]
    return pc


  def render_worlds_frame_homogenious_pc(self):
    pc = self.render_camera_frame_pc()
    _, _, rotation, translation = self.camera.matrices()
    homgenious_pc = np.concatenate((pc, np.ones((pc.shape[0], 1), dtype=np.float32)), axis=1)
    return ( -translation @ np.linalg.inv(rotation) @ homgenious_pc.T).T


  def render_worlds_frame_pc(self):
    return self.render_worlds_frame_homogenious_pc()[:, :3]

  
  def save_kitti_pc(self, filename):
    data = self.render_worlds_frame_homogenious_pc()
    assert(data.shape[1] == 4)
    print(data.shape)
    data[:,3] = 1
    data = data.astype(np.float32)
    f = open(filename, "wb")
    for e in data:
      f.write(e.ravel())
    f.close()