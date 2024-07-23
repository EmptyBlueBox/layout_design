import os
import mujoco
import numpy
import mediapy as media
import matplotlib.pyplot as plt
import math
import yaml
import scipy.interpolate
import mujoco.viewer as viewer

xml_template = """
<mujoco>
  <option timestep="{ts:e}"/>
  <option integrator="{integrator}"/>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="A" pos="0 0 0">
            <joint name="j1" type="hinge" axis="0 1 0" pos="0 0 0"/>
            <geom type="box" size=".5 .05 .05" pos=".5 0 0" rgba="1 0 0 1" mass="1"/>
            <body name="B" pos="1 0 0">
                <joint name="j2" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom type="box" size=".5 .05 .05" pos=".5 0 0" rgba="1 0 0 1" mass="1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <general name="m1" joint="j1"/>
        <general name="m2" joint="j2"/>
    </actuator>
</mujoco>
"""
xml = xml_template.format(ts=1e-3, integrator='RK4')

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

freq = 1


def my_controller(model, data):
    theta1 = math.sin(2*math.pi*(freq*data.time))
    theta2 = math.sin(2*math.pi*(freq*data.time-.25))
    data.ctrl = [theta1, theta2]
    return


mujoco.mj_resetData(model, data)

duration = 3
framerate = 30

q = []
w = []
a = []
t = []
xy = []
frames = []

try:
    mujoco.set_mjcb_control(my_controller)
    while data.time < duration:

        mujoco.mj_step(model, data)

        q.append(data.qpos.copy())
        w.append(data.qvel.copy())
        a.append(data.qacc.copy())
        xy.append(data.xpos.copy())
        t.append(data.time)

        if len(frames) < data.time*framerate:

            renderer.update_scene(data)
            # renderer.update_scene(data,"world")
            pixels = renderer.render()
            frames.append(pixels)
finally:
    mujoco.set_mjcb_control(None)

# media.show_video(frames, fps=framerate)
# 保存视频
media.write_video('./imgs/inverse_dynamics.mp4', frames, fps=framerate)

q = numpy.array(q)
w = numpy.array(w)
a = numpy.array(a)
xy = numpy.array(xy)
t = numpy.array(t)

plt.plot(t, q)
plt.savefig('./imgs/position.png')
plt.close()

plt.plot(t, w)
plt.savefig('./imgs/velocity.png')
plt.close()

plt.plot(t, a)
plt.savefig('./imgs/acceleration.png')
plt.close()

mujoco.mj_resetData(model, data)
torque_est = []

for q_ii, w_ii, a_ii in zip(q, w, a):
    data.qpos[:] = q_ii
    data.qvel[:] = w_ii
    data.qacc[:] = a_ii
    mujoco.mj_inverse(model, data)
    torque = data.qfrc_inverse.copy()
    torque_est.append(torque)

torque_est = numpy.array(torque_est)

plt.plot(torque_est)
# 保存图片
plt.savefig('./imgs/torque_est.png')
