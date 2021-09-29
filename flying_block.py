import numpy as np
from dm_control import mjcf
import random

class FlyingBlock():
  def __init__(self, idx, body_radius=0.1):
    self.idx = idx
    self.name = f"block_{idx:06d}"
    self.model = mjcf.RootElement()
    self.model.model = self.name


    
    body_radius = body_radius
    body_height = body_radius / 2
    body_size = (body_radius, body_radius, body_height)

    rgba = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])
    self.model.worldbody.add(
        'geom', name='body', type='ellipsoid', size=body_size, rgba=rgba)    

  def add_to_arena(self, arena, spawn_pos):
    spawn_site = arena.worldbody.add('site', pos=spawn_pos)
    spawn_site.attach(self.model).add('freejoint')

  def actuators(self):
    return self.model.find_all('actuator')

  def initialize_velocities(self, physics, seed):
    rng = random.Random(seed + self.idx)
    x = rng.uniform(-2, 2)
    y = rng.uniform(-2, 2)
    z = rng.uniform(1, 5)
    vals = physics.named.data.qvel[self.name + "/"]
    vals[0] = x
    vals[1] = y
    vals[2] = z
