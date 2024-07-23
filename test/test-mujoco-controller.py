import time
import mujoco
import mujoco.viewer
import math

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

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

freq = 1


def controller(model, data):
    theta1 = math.sin(2*math.pi*(freq*data.time))
    theta2 = math.sin(2*math.pi*(freq*data.time-.25))
    data.ctrl = [theta1, theta2]
    return


mujoco.set_mjcb_control(controller)


with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
