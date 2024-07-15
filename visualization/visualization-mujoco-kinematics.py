'''
mjpython visualization-mujoco-kinematics.py

1. Call metric.metric_torque.get_mujoco_data(data_name) to get motion data
2. Use passive viewer to visualize the motion data without physics simulation
'''
import time
import mujoco
import mujoco.viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from metric.metric_torque import get_mujoco_data, set_mujoco_data, get_best_z_offset
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('../humanoid/smplx_humanoid-only_body.xml')
data = mujoco.MjData(model)

data_name = 'seat_1-frame_num_150'
motion_data = get_mujoco_data(data_name)
fps = motion_data['fps']
total_frame_num = motion_data['frame_num']

best_offset = get_best_z_offset(model, motion_data)

# 可视化
mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    frame_num = 0
    while viewer.is_running() and time.time() - start < 30:
        print(f'frame_num: {frame_num}')
        step_start = time.time()

        mujoco.mj_step(model, data)

        set_mujoco_data(data, motion_data, frame_num, best_offset)
        set_mujoco_data(data, motion_data, 38, best_offset)

        viewer.sync()

        time_until_next_step = 1.0 / fps * 5 - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        frame_num += 1
        if frame_num == total_frame_num:
            break
