import time
import mujoco
import mujoco.viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from metric.metric_torque import get_mujoco_data

model = mujoco.MjModel.from_xml_path('../humanoid/smplx_humanoid-only_body.xml')
model.opt.gravity = (0, -9.81, 0)
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

torque_estimation = np.zeros((human_root_position.shape[0], model.nv))
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    frame_num = 0
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        mujoco.mj_step(model, data)

        # 位置
        data.qpos[0:3] = human_root_position[frame_num, 0:3].copy()
        data.qvel[0:3] = human_root_velocity[frame_num, 0:3].copy()
        data.qacc[0:3] = human_root_acceleration[frame_num, 0:3].copy()
        print(f'counter: {frame_num}, human_root_position: {human_root_position.shape}')
        for j in range(human_pose_euler.shape[1]):  # 0-21
            # 角度
            if j == 0:
                human_quat = R.from_euler('xyz', human_pose_euler[frame_num, j]).as_quat()
                data.qpos[3:7] = human_quat[[3, 0, 1, 2]].copy()
            else:
                data.qpos[4+3*j:4+3*j+3] = human_pose_euler[frame_num, j].copy()
            # 角速度
            data.qvel[3+3*j:3+3*j+3] = human_pose_angular_velocity[frame_num, j].copy()
            # 角加速度
            data.qacc[3+3*j:3+3*j+3] = human_pose_angular_acceleration[frame_num, j].copy()

        viewer.sync()

        with viewer.lock():
            mujoco.mj_inverse(model, data)
            torque_estimation[frame_num-1] = data.qfrc_inverse.copy()

        time_until_next_step = 1.0 / fps - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        frame_num += 1
        if frame_num == human_root_position.shape[0]:
            break

torque_left_thigh = torque_estimation[:, 3+3*1:3+3*2].reshape(-1, 3)
print(f'torque_left_thigh shape: {torque_left_thigh.shape}')
print(f'torque_left_thigh: {np.max(torque_left_thigh[:,0])}')
print(f'torque_left_thigh: {np.max(torque_left_thigh[:,1])}')
print(f'torque_left_thigh: {np.max(torque_left_thigh[:,2])}')
# plt.plot(torque_left_thigh[:, 0])
# plt.plot(torque_left_thigh[:, 1])
# plt.plot(torque_left_thigh[:, 2])
# plt.savefig(f'./imgs/torque_left_thigh-{data_name}.png')
# plt.close()
