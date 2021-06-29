import numpy as np
from dm_control import mjcf

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

class Spider():
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

    # Attach legs to equidistant sites on the circumference.
    for i in range(num_legs):
      theta = 2 * i * np.pi / num_legs
      hip_pos = body_radius * np.array([np.cos(theta), np.sin(theta), 0])
      hip_site = model.worldbody.add('site', pos=hip_pos, euler=[0, 0, np.rad2deg(theta)])
      leg = Leg(length=body_radius, rgba=rgba)
      hip_site.attach(leg.model)

    self.model = model

  def add_to_arena(self, arena, spawn_pos):
    spawn_site = arena.worldbody.add('site', pos=spawn_pos, group=3)
    spawn_site.attach(self.model).add('freejoint')

  def actuators(self):
    return self.model.find_all('actuator')