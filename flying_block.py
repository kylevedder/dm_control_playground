import numpy as np
from dm_control import mjcf

class FlyingBlock():
  def __init__(self, body_radius=0.1, num_legs=3):

    """Constructs a creature with `num_legs` legs."""
    rgba = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])
    model = mjcf.RootElement()
    # model.compiler.angle = 'radian'  # Use radians.

    self.body_radius = body_radius
    self.body_height = body_radius / 2
    body_size = (self.body_radius, self.body_radius, self.body_height)
    # Make the torso geom.
    model.worldbody.add(
        'geom', name='torso', type='ellipsoid', size=body_size, rgba=rgba)

    self.model = model

  def add_to_arena(self, arena, spawn_pos):
    spawn_site = arena.worldbody.add('site', pos=spawn_pos, group=3)
    spawn_site.attach(self.model).add('freejoint')

  def actuators(self):
    return self.model.find_all('actuator')