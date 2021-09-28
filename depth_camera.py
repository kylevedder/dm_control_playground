import cv2
import numpy as np
from PIL import Image
from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import joblib

class DepthCamera:
  def __init__(self, camera):
    self.camera = camera
    self.video_frames = []


  def render_depth(self):
    return self.camera.render(depth=True)
  
  def render_img(self):
    return self.camera.render()

  def render_video_frame(self):
    self.video_frames.append(self.render_img().copy())

  def render_segmentation(self):
    sem = self.camera.render(segmentation=True)
    img = np.zeros((*sem.shape[0:2], 3))
    types = sem[:, :, 0]
    ids = sem[:, :, 1]
    max_val = max(np.unique(types))
    for t in np.unique(types):
      if t <= 0:
        continue
      img[types == t, 0] = t / max_val
    return (img * 255).astype(np.uint8)

  def render_depth_minus_background(self):
    depth_img = self.render_depth()
    types = self.camera.render(segmentation=True)[:,:,0]
    depth_img[types <= 0] = np.nan
    return depth_img


  def save_video(self, filename, framerate=30):
    import matplotlib
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    height, width, _ = self.video_frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(self.video_frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=self.video_frames,
                                   interval=interval, blit=True, repeat=False)
    return anim.save(filename)

  def render_camera_frame_pc(self, background_subtract=False):
    if background_subtract:
      depth_img = self.render_depth_minus_background()
    else:
      depth_img = self.render_depth()
    image, focal, _, _ = self.camera.matrices()
    pc = cv2.rgbd.depthTo3d(depth_img, image @ focal[:, :3])    
    types = self.camera.render(segmentation=True)[:,:,0]
    pc = np.concatenate([pc, np.expand_dims(types, 2)], 2)
    pc = pc.reshape(pc.shape[0] * pc.shape[1], 4)
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

  def save_ply(self, filename, background_subtract=False):
    points = self.render_camera_frame_pc(background_subtract=background_subtract)
    PyntCloud(pd.DataFrame(data=points[:, :3],
        columns=["x", "y", "z"])).to_file(filename)


  def save_np_and_classes(self, np_filename, classes_filename, background_subtract=False):
    points_and_classes = self.render_camera_frame_pc(background_subtract=background_subtract)
    points = points_and_classes[:, :3]
    classes = (points_and_classes[:, 3] - 1) // 7
    joblib.dump(points, np_filename)
    joblib.dump(classes, classes_filename)


  def save_open3d_pcd_and_classes(self, pcd_filename, classes_filename, background_subtract=False):
    points_and_classes = self.render_camera_frame_pc(background_subtract=background_subtract)
    points = points_and_classes[:, :3]
    # Each spider is made of 7 objects, and - 1 removes the ground plane
    classes = (points_and_classes[:, 3] - 1) // 7
    o3d.io.write_point_cloud(pcd_filename, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    joblib.dump(classes, classes_filename)




  def save_img(self, filename):
    img = self.render_img()
    im = Image.fromarray(img)
    im.save(filename)

  def save_segmentation(self, filename):
    img = self.render_segmentation()
    im = Image.fromarray(img)
    im.save(filename)



