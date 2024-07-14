import mujoco
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from utils.data_utils import butterworth_filter

fps = 30
desired_fps = 1/0.002
desired_fps = 30
smplx2mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]


def get_mujoco_data(data_name):
    # 读取human_params数据
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{data_name}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    frame_num = human_params['poses'].shape[0]

    # 计算人体的根结点的位置
    human_root_position = human_params['translation']
    x = np.linspace(0, frame_num/fps, int(frame_num/fps*desired_fps))  # 从0到frame_num/FPS, 生成frame_num/FPS*desired_fps个数
    xp = np.linspace(0, frame_num/fps, frame_num)
    xf = human_root_position
    human_root_position_interp = np.array([np.interp(x, xp, xf[:, 0]), np.interp(x, xp, xf[:, 1]), np.interp(x, xp, xf[:, 2])]).T
    for i in range(3):
        human_root_position_interp[:, i] = butterworth_filter(human_root_position_interp[:, i], cutoff=1, fs=desired_fps)
    human_root_position = human_root_position_interp
    human_root_position = human_root_position[:, [0, 2, 1]]  # 转换为 Mujoco 的坐标系, z up
    human_root_position[:, 1] *= -1  # y轴取反, 因为旋转后的y是之前的-z
    # 计算人体的根结点的速度
    human_root_velocity = np.gradient(human_root_position, axis=0) * desired_fps
    # 计算人体的根结点的加速度
    human_root_acceleration = np.gradient(human_root_velocity, axis=0) * desired_fps

    # 计算人体的欧拉角
    human_rotation_euler = np.zeros_like(human_params['poses'])
    for i in range(human_params['poses'].shape[0]):
        # 将旋转向量转换为欧拉角
        human_rotation_euler[i] = R.from_rotvec(human_params['poses'][i]).as_euler('xyz', degrees=False)

    # 计算root的欧拉角
    human_orientation_euler = np.zeros_like(human_params['orientation'])
    human_orientation_euler = R.from_rotvec(human_params['orientation']).as_euler('xyz', degrees=False)

    # 转换为 Mujoco 的坐标系, z up
    rot = R.from_rotvec([np.pi/2, 0, 0])
    human_orientation_euler_z_up = (rot*R.from_euler('xyz', human_orientation_euler, degrees=False)).as_euler('xyz', degrees=False)
    human_orientation_euler = human_orientation_euler_z_up

    # 将人体的欧拉角和root的欧拉角合并, 并将其转换为 Mujoco 的顺序
    human_pose_euler = np.concatenate([human_orientation_euler[:, None, :], human_rotation_euler], axis=1)[:, smplx2mujoco]
    print(f'human_euler shape: {human_pose_euler.shape}')

    # Slerp 插值
    human_pose_euler_interp = np.zeros((x.shape[0], human_pose_euler.shape[1], human_pose_euler.shape[2]))
    for i in range(human_pose_euler.shape[1]):
        slerp = Slerp(xp, R.from_euler(seq='xyz', angles=human_pose_euler[:, i], degrees=False))
        human_pose_euler_interp[:, i] = slerp(x).as_euler('xyz', degrees=False)
    for i in range(human_pose_euler_interp.shape[1]):
        for j in range(3):
            human_pose_euler_interp[:, i, j] = butterworth_filter(human_pose_euler_interp[:, i, j], cutoff=1, fs=desired_fps)
    # human_pose_euler = human_pose_euler_interp

    # 计算角速度
    human_pose_angular_velocity = np.gradient(human_pose_euler, axis=0) * desired_fps
    print(f'human_pose_angular_velocity shape: {human_pose_angular_velocity.shape}')

    # 计算角加速度
    human_pose_angular_acceleration = np.gradient(human_pose_angular_velocity, axis=0) * desired_fps
    print(f'human_pose_angular_acceleration shape: {human_pose_angular_acceleration.shape}')

    output = {
        'fps': desired_fps,
        'human_root_position': human_root_position,
        'human_root_velocity': human_root_velocity,
        'human_root_acceleration': human_root_acceleration,
        'human_pose_euler': human_pose_euler,
        'human_pose_angular_velocity': human_pose_angular_velocity,
        'human_pose_angular_acceleration': human_pose_angular_acceleration
    }
    return output


def plot(data: dict):
    human_root_position = data['human_root_position']
    human_root_velocity = data['human_root_velocity']
    human_root_acceleration = data['human_root_acceleration']
    human_pose_euler = data['human_pose_euler']
    human_pose_angular_velocity = data['human_pose_angular_velocity']
    human_pose_angular_acceleration = data['human_pose_angular_acceleration']

    plt.plot(human_pose_euler[:, 1, 0], label='x')
    plt.plot(human_pose_euler[:, 1, 1], label='y')
    plt.plot(human_pose_euler[:, 1, 2], label='z')
    plt.title('Human Euler Left Thigh')
    plt.legend()
    plt.savefig('./imgs/human_euler-left_thigh.png')
    plt.close()

    plt.plot(human_pose_angular_velocity[:, 1, 0], label='x')
    plt.plot(human_pose_angular_velocity[:, 1, 1], label='y')
    plt.plot(human_pose_angular_velocity[:, 1, 2], label='z')
    plt.title('Human Angular Velocity Left Thigh')
    plt.legend()
    plt.savefig('./imgs/human_angular_velocity-left_thigh.png')
    plt.close()

    plt.plot(human_pose_angular_acceleration[:, 1, 0], label='x')
    plt.plot(human_pose_angular_acceleration[:, 1, 1], label='y')
    plt.plot(human_pose_angular_acceleration[:, 1, 2], label='z')
    plt.title('Human Angular Acceleration Left Thigh')
    plt.legend()
    plt.savefig('./imgs/human_angular_acceleration-left_thigh.png')
    plt.close()


# # 可视化模型
# viewer.launch(model, data)
# exit(0)

def get_torque(motion_data: dict):
    # 从XML字符串创建MjModel对象
    model = mujoco.MjModel.from_xml_path('../humanoid/smplx_humanoid-only_body.xml')
    model.opt.gravity = (0, -9.8, 0)
    # model.opt.timestep = 1/FPS
    print(f'default timestep: {model.opt.timestep}')
    # model.opt.disableflags = 1 << 4 + 1 << 1  # disable contact constraints
    # model.opt.integrator = 0  # change integrator to Euler

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

    mujoco.mj_resetData(model, data)

    human_root_position = motion_data['human_root_position']
    human_root_velocity = motion_data['human_root_velocity']
    human_root_acceleration = motion_data['human_root_acceleration']
    human_pose_euler = motion_data['human_pose_euler']
    human_pose_angular_velocity = motion_data['human_pose_angular_velocity']
    human_pose_angular_acceleration = motion_data['human_pose_angular_acceleration']

    torque_est = []
    for i in range(human_root_position.shape[0]):
        # 位置
        data.qpos[0:3] = human_root_position[i, 0:3].copy()
        # 速度
        data.qvel[0:3] = human_root_velocity[i, 0:3].copy()
        # 加速度
        data.qacc[0:3] = human_root_acceleration[i, 0:3].copy()
        for j in range(human_pose_euler.shape[1]):  # 0-21
            # 角度
            if j == 0:
                human_quat = R.from_euler('xyz', human_pose_euler[i, j]).as_quat()
                data.qpos[3:7] = human_quat[[3, 0, 1, 2]].copy()
            else:
                data.qpos[4+3*j:4+3*j+3] = human_pose_euler[i, j].copy()
            # 角速度
            data.qvel[3+3*j:3+3*j+3] = human_pose_angular_velocity[i, j].copy()
            # 角加速度
            data.qacc[3+3*j:3+3*j+3] = human_pose_angular_acceleration[i, j].copy()
        mujoco.mj_inverse(model, data)
        torque = data.qfrc_inverse.copy()
        torque_est.append(torque)
    torque_est = np.array(torque_est)
    print(f'torque_est shape: {torque_est.shape}')
    return torque_est


def print_torque(torque_est):
    torque_left_thigh = torque_est[:, 3+3*1:3+3*2].reshape(-1, 3)
    plt.plot(torque_left_thigh[:, 0], label='x')
    plt.plot(torque_left_thigh[:, 1], label='y')
    plt.plot(torque_left_thigh[:, 2], label='z')
    plt.title('Torque Left Thigh')
    plt.legend()
    plt.savefig('./imgs/torque_left_thigh.png')
    plt.close()

    torque_right_thigh = torque_est[:, 3+3*5:3+3*6].reshape(-1, 3)
    plt.plot(torque_right_thigh[:, 0], label='x')
    plt.plot(torque_right_thigh[:, 1], label='y')
    plt.plot(torque_right_thigh[:, 2], label='z')
    plt.title('Torque Right Thigh')
    plt.legend()
    plt.savefig('./imgs/torque_right_thigh.png')
    plt.close()


def main():
    data_name = 'seat_5-frame_num_150'
    data = get_mujoco_data(data_name)
    plot(data)
    torque_est = get_torque(data)
    print_torque(torque_est)
    print('Done')


if __name__ == '__main__':
    main()
