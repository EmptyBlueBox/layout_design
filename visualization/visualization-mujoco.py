import time
import mujoco
import mujoco.viewer
import mujoco
import mediapy as media
import numpy as np
from PIL import Image
import pickle
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('../humanoid/smplx_humanoid-only_body.xml')
model.opt.gravity = (0, -9.81, 0)
data = mujoco.MjData(model)

data_name = 'seat_5-frame_num_150'
DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{data_name}'
human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))

FPS = 30
smplx2mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]

# 计算人体的根结点的位置, 速度和加速度
human_root_position = human_params['translation']
human_root_velocity = np.gradient(human_root_position, axis=0) * FPS
human_root_acceleration = np.gradient(human_root_velocity, axis=0) * FPS

# 计算人体的欧拉角
human_pose_euler = np.zeros_like(human_params['poses'])
for i in range(human_params['poses'].shape[0]):
    # 将旋转向量转换为欧拉角
    for j in range(human_params['poses'].shape[1]):
        human_pose_euler[i, j] = R.from_rotvec(human_params['poses'][i, j]).as_euler('xyz', degrees=False)

# 计算root的欧拉角
human_orientation_euler = np.zeros_like(human_params['orientation'])
for i in range(human_params['poses'].shape[0]):
    human_orientation_euler[i] = R.from_rotvec(human_params['orientation'][i]).as_euler('xyz', degrees=False)

# 将人体的欧拉角和root的欧拉角合并, 并将其转换为Mujoco的顺序
human_euler = np.concatenate([human_orientation_euler[:, None, :], human_pose_euler], axis=1)[:, smplx2mujoco]
print(f'human_euler shape: {human_euler.shape}')

# 计算角速度
human_pose_angular_velocity = np.zeros_like(human_euler)
human_pose_angular_velocity = np.gradient(human_euler, axis=0) * FPS
print(f'human_pose_angular_velocity shape: {human_pose_angular_velocity.shape}')

# 计算角加速度
human_pose_angular_acceleration = np.zeros_like(human_pose_angular_velocity)
human_pose_angular_acceleration = np.gradient(human_pose_angular_velocity, axis=0) * FPS
print(f'human_pose_angular_acceleration shape: {human_pose_angular_acceleration.shape}')

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
        for j in range(human_euler.shape[1]):  # 0-21
            # 角度
            if j == 0:
                human_quat = R.from_euler('xyz', human_euler[frame_num, j]).as_quat()
                data.qpos[3:7] = human_quat[[3, 0, 1, 2]].copy()
            else:
                data.qpos[4+3*j:4+3*j+3] = human_euler[frame_num, j].copy()
            # 角速度
            data.qvel[3+3*j:3+3*j+3] = human_pose_angular_velocity[frame_num, j].copy()
            # 角加速度
            data.qacc[3+3*j:3+3*j+3] = human_pose_angular_acceleration[frame_num, j].copy()

        viewer.sync()

        with viewer.lock():
            mujoco.mj_inverse(model, data)
            torque_estimation[frame_num-1] = data.qfrc_inverse.copy()

        time_until_next_step = 1.0 / FPS - (time.time() - step_start)
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
