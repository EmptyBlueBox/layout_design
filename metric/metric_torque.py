import mujoco
import mujoco.viewer as viewer
import mediapy as media
import numpy as np
from PIL import Image
import pickle
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

FPS = 30
smplx2mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
# 读取XML模型文件
with open('../humanoid/smplx_humanoid-only_body.xml', 'r') as f:
    xml = f.read()

# 从XML字符串创建MjModel对象
model = mujoco.MjModel.from_xml_string(xml)
model.opt.timestep = 0.002  # default

data = mujoco.MjData(model)
print([model.body(i).name for i in range(model.ngeom)])
print(f'Number of geoms in the model: {model.ngeom}')
print(f'Number of joints in the model: {model.njnt}')
print(f'Number of degrees of freedom in the model: {model.nv}')
print(f'Number of bodies in the model: {model.nbody}')

print(f'q shape: {data.qpos.shape}')
print(f'w shape: {data.qvel.shape}')
print(f'a shape: {data.qacc.shape}')
print(f'xy shape: {data.xpos.shape}')

# 读取human_params数据
data_name = 'seat_1-frame_num_150'
DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{data_name}'
human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))

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

plt.plot(human_euler[:, 1, 0])
plt.plot(human_euler[:, 1, 1])
plt.plot(human_euler[:, 1, 2])
plt.savefig('./imgs/human_euler-left_thigh.png')
plt.close()

plt.plot(human_pose_angular_velocity[:, 1, 0])
plt.plot(human_pose_angular_velocity[:, 1, 1])
plt.plot(human_pose_angular_velocity[:, 1, 2])
plt.savefig('./imgs/human_angular_velocity-left_thigh.png')
plt.close()

plt.plot(human_pose_angular_acceleration[:, 1, 0])
plt.plot(human_pose_angular_acceleration[:, 1, 1])
plt.plot(human_pose_angular_acceleration[:, 1, 2])
plt.savefig('./imgs/human_angular_acceleration-left_thigh.png')
plt.close()


# # 可视化模型
# def visualize_model(model, data):
#     # 使用Mujoco的viewer进行可视化
#     viewer.launch(model, data)
# # 调用可视化函数
# visualize_model(model, data)

mujoco.mj_resetData(model, data)
torque_est = []
for i in range(150):
    # data.xpos[1] = human_params['translation'][i, 0]
    for j in range(human_euler.shape[1]):  # 0-21
        data.qpos[1+3+3*j] = human_euler[i, j, 0]
        data.qpos[1+3+3*j+1] = human_euler[i, j, 1]
        data.qpos[1+3+3*j+2] = human_euler[i, j, 2]
        data.qvel[3+3*j] = human_pose_angular_velocity[i, j, 0]
        data.qvel[3+3*j+1] = human_pose_angular_velocity[i, j, 1]
        data.qvel[3+3*j+2] = human_pose_angular_velocity[i, j, 2]
        data.qacc[3+3*j] = human_pose_angular_acceleration[i, j, 0]
        data.qacc[3+3*j+1] = human_pose_angular_acceleration[i, j, 1]
        data.qacc[3+3*j+2] = human_pose_angular_acceleration[i, j, 2]
    mujoco.mj_inverse(model, data)
    torque = data.qfrc_inverse.copy()
    torque_est.append(torque)
torque_est = np.array(torque_est)
print(f'torque_est shape: {torque_est.shape}')

torque_left_thigh = torque_est[:, 3+3*2:3+3*3].reshape(-1, 3)
plt.plot(torque_left_thigh[:, 0])
plt.plot(torque_left_thigh[:, 1])
plt.plot(torque_left_thigh[:, 2])
plt.savefig('./imgs/torque_left_thigh.png')
plt.close()

torque_right_thigh = torque_est[:, 3+3*6:3+3*7].reshape(-1, 3)
plt.plot(torque_right_thigh[:, 0])
plt.plot(torque_right_thigh[:, 1])
plt.plot(torque_right_thigh[:, 2])
plt.savefig('./imgs/torque_right_thigh.png')
plt.close()
