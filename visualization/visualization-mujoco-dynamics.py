'''
mjpython visualization-mujoco-dynamics.py

1. Call metric.metric_torque.get_mujoco_data(data_name) to get motion data
2. Call metric.metric_torque.get_torque(motion_data) to get torque estimation
3. Use passive viewer to visualize the motion data with physics simulation
'''
import time
import mujoco
import mujoco.viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from metric.metric_torque import get_mujoco_data, get_torque

model = mujoco.MjModel.from_xml_path('../humanoid/smplx_humanoid-only_body.xml')
data = mujoco.MjData(model)

data_name = 'seat_5-frame_num_150'
motion_data = get_mujoco_data(data_name)

fps = motion_data['fps']
human_root_position = motion_data['human_root_position']  # (frame_num, 3)
human_root_velocity = motion_data['human_root_velocity']  # (frame_num, 3)
human_root_acceleration = motion_data['human_root_acceleration']  # (frame_num, 3)
human_pose_euler = motion_data['human_pose_euler']  # (frame_num, 22, 3)
human_pose_angular_velocity = motion_data['human_pose_angular_velocity']  # (frame_num, 22, 3)
human_pose_angular_acceleration = motion_data['human_pose_angular_acceleration']  # (frame_num, 22, 3)

torque_estimation = get_torque(motion_data)


def controller_torque(model, data):
    fps = motion_data['fps']
    index = int(data.time * fps)  # frame_num
    data.ctrl = torque_estimation[index, 6:]
    return


# PD控制器增益
Kp = 10.0  # 比例增益
Kd = 1.0   # 微分增益


def controller_pd(model, data):
    fps = motion_data['fps']
    index = int(data.time * fps)  # frame_num

    return


mujoco.set_mjcb_control(controller_pd)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    frame_num = 0
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = 1.0 / fps - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        frame_num += 1
        if frame_num == human_root_position.shape[0]:
            break
