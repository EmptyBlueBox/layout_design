import pickle
import smplx
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.data_utils import butterworth_filter
import mujoco
from metric.metric_torque import get_mujoco_data, set_mujoco_data, get_best_z_offset

Head = 8.23
Thorax = 18.56
Abdomen = 12.65
Pelvis = 14.81
Upper_Arm = 3.075
Forearm = 1.72
Hand = 0.575
Forearm_Hand = 2.295
Thigh = 11.125
Leg = 5.05
Foot = 1.38
Leg_Foot = 6.43

'''
12: neck
13: left thorax
14: right thorax
15: head
'''
MASS_DISTRIBUTION = np.array([Pelvis, Thigh/2, Thigh/2, Abdomen/2, Leg/2, Leg/2, Abdomen/2, Foot/4, Foot/4, Thorax/4,
                             Foot/4, Foot/4, Thorax/4, Thorax/4, Thorax/4, Head, Upper_Arm/2, Upper_Arm/2, Forearm/2,
                             Forearm/2, Hand/2, Hand/2])
FPS = 30


def mechanical_energy_vanilla(human_params, FPS=30, data_name=None):
    """
    This Python function calculates various aspects of mechanical energy, including translational and
    rotational energy, for a human model based on input parameters.

    Arguments:

    * `human_params`: It seems like the definition of the function `mechanical_energy` is missing some
    key information about the `human_params` that are required for the function to run successfully.
    Could you please provide the details of the `human_params` dictionary that is passed as an argument
    to the function? This will
    * `FPS`: Frames per second (FPS) is a parameter that defines the number of frames displayed or
    processed per second in a video or animation. In the provided code snippet, the default value for
    FPS is set to 30 frames per second. This means that the calculations and analysis in the
    `mechanical_energy
    * `data_name`: `data_name` is a parameter that can be passed to the `mechanical_energy` function to
    specify the name of the data being processed. It is an optional parameter and can be used to provide
    a descriptive label for the data being analyzed.

    Returns:

    The function `mechanical_energy` returns a dictionary `output` containing the following key-value
    pairs: 'data_name', 'potential_energy', 'translational_energy', 'rotational_energy', 'filtered_kinetic_energy',
    'mechanical_energy', 'filtered_mechanical_energy', 'power', 'velocity', 'angular_velocity', and 'filtered_angular_velocity'.
    """
    FRAME_NUM = human_params['poses'].shape[0]  # 100
    human_model = smplx.create(model_path='/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=FRAME_NUM)
    output = human_model(body_pose=torch.tensor(human_params['poses'], dtype=torch.float32),
                         global_orient=torch.tensor(human_params['orientation'], dtype=torch.float32),
                         transl=torch.tensor(human_params['translation'], dtype=torch.float32))
    key_points = output.joints[:, :22].detach().numpy()  # (100, 22, 3), 所有骨骼关键点的坐标

    # 计算平动速度
    velocity = []
    for i in range(FRAME_NUM):
        if i == 0:
            velocity.append(np.zeros((22, 3)))
        else:
            velocity.append((key_points[i] - key_points[i-1]) * FPS)
        for j in range(1, 22):
            velocity[i][j] = (velocity[i][j]+velocity[i][human_model.parents[j]])/2
    velocity = np.array(velocity)  # (150, 22, 3)
    velocity = np.linalg.norm(velocity, axis=2)  # (150, 22)

    # 计算角速度
    angular_velocity = []
    for i in range(FRAME_NUM):
        if i == 0:
            angular_velocity.append(np.zeros((22)))
        else:
            # 计算角速度
            one_frame_angular_velocity = []
            root_former_rot = R.from_rotvec(human_params['orientation'][i-1])
            root_now_rot = R.from_rotvec(human_params['orientation'][i])
            root_delta_rot = root_now_rot*root_former_rot.inv()
            one_frame_angular_velocity.append(np.linalg.norm(root_delta_rot.as_rotvec())*FPS)
            for j in range(1, 22):
                former_rot = R.from_rotvec(human_params['poses'][i-1, j-1])
                now_rot = R.from_rotvec(human_params['poses'][i, j-1])
                delta_rot = now_rot*former_rot.inv()
                one_frame_angular_velocity.append(np.linalg.norm(delta_rot.as_rotvec())*FPS)
            angular_velocity.append(one_frame_angular_velocity)  # (150, 22)
    angular_velocity = np.array(angular_velocity)  # (150, 22)

    # 计算平均骨骼长度
    bone_length = []
    for i in range(FRAME_NUM):
        one_frame_bone_length = [0]
        for j in range(1, 22):
            one_frame_bone_length.append(np.linalg.norm(key_points[i, j] - key_points[i, human_model.parents[j]]))  # cm
        bone_length.append(one_frame_bone_length)  # (150, 22)
    bone_length = np.mean(np.array(bone_length), axis=0)

    # 计算转动惯量
    MOMENT_OF_INERTIA = np.array([MASS_DISTRIBUTION * bone_length ** 2 / 12])  # (22,)

    # 计算每一帧的能量
    potential_energy = []
    translational_energy = []
    rotational_energy = []
    key_points -= key_points[0]
    for i in range(FRAME_NUM):
        # 计算每一帧的机械能
        potential_energy.append(np.sum(MASS_DISTRIBUTION * key_points[i, :, 1] * 9.81))  # 相对势能
        translational_energy.append(np.sum(MASS_DISTRIBUTION * velocity[i] ** 2) / 2)  # 平动动能
        rotational_energy.append(np.sum(MOMENT_OF_INERTIA * angular_velocity[i] ** 2) / 2)  # 旋转动能

    # 转换为numpy数组
    potential_energy = np.array(potential_energy)
    translational_energy = np.array(translational_energy)
    rotational_energy = np.array(rotational_energy)

    # 计算总机械能
    mechanical_energy = potential_energy + translational_energy + rotational_energy

    # 计算总做功下界
    power = np.zeros_like(mechanical_energy)  # 人体做功功率下界
    for i in range(1, FRAME_NUM):
        if mechanical_energy[i] > mechanical_energy[i-1]:
            power[i] = mechanical_energy[i] - mechanical_energy[i-1]  # 机械能增加的部分

    move_dis = human_params['translation'][0, [0, 2]]-human_params['translation'][-1, [0, 2]]  # (2,)
    print(f'move distance: {np.linalg.norm(move_dis)}')

    filtered_angular_velocity = np.zeros_like(angular_velocity)
    for i in range(22):
        filtered_angular_velocity[:, i] = butterworth_filter(angular_velocity[:, i])

    output = {
        'data_name': data_name,
        'potential_energy': potential_energy,
        'translational_energy': translational_energy,
        'rotational_energy': rotational_energy,
        'filtered_kinetic_energy': butterworth_filter(translational_energy + rotational_energy, 5),
        'mechanical_energy': mechanical_energy,  # 'energy': 'potential_energy + kinetic_energy + rotational_energy
        'filtered_mechanical_energy': butterworth_filter(mechanical_energy, 5),
        'power': power,
        'velocity': velocity,
        'angular_velocity': angular_velocity,
        'filtered_angular_velocity': filtered_angular_velocity,
    }

    return output


def mechanical_energy_mujoco(data_name=None):
    model = mujoco.MjModel.from_xml_path('../humanoid/smplx_humanoid-only_body.xml')
    print(f'default timestep: {model.opt.timestep}')
    model.opt.timestep = 1e-4
    model.opt.disableflags = 1 << 4  # disable contact constraints
    # model.opt.integrator = 0  # change integrator to Euler

    data = mujoco.MjData(model)

    motion_data = get_mujoco_data(data_name, 5)
    motion_data['human_root_position'] -= motion_data['human_root_position'][0]  # 重力势能一样

    potential_energy = []
    kinetic_energy = []
    mujoco.mj_resetData(model, data)
    for i in range(motion_data['frame_num']):
        mujoco.mj_resetData(model, data)
        set_mujoco_data(data, motion_data, i, static=False)
        # data.qvel = 0  # 为什么加上速度, 因为加速度会变大很多, 从而动能会非常高
        mujoco.mj_step(model, data)
        potential_energy.append(data.energy[0])
        kinetic_energy.append(data.energy[1])
    potential_energy = np.array(potential_energy)
    kinetic_energy = np.array(kinetic_energy)  # (150,)
    mechanical_energy = potential_energy + kinetic_energy  # (150,)
    power = np.gradient(mechanical_energy)  # (150,)

    output = {
        'data_name': data_name,
        'potential_energy': potential_energy,
        'translational_energy': kinetic_energy,
        'rotational_energy': kinetic_energy,
        'filtered_kinetic_energy': butterworth_filter(kinetic_energy, 5),
        'mechanical_energy': mechanical_energy,  # 'energy': 'potential_energy + kinetic_energy + rotational_energy
        'filtered_mechanical_energy': butterworth_filter(mechanical_energy, 5),
        'power': power,
    }
    return output


def plot_energy(data: list, data_type: str):
    # for data_entry in data:
    #     # 绘制能量
    #     plt.figure()
    #     plt.plot(data_entry['potential_energy'], label='potential energy')
    #     plt.plot(data_entry['translational_energy'], label='translational energy')
    #     plt.plot(data_entry['rotational_energy'], label='rotational energy')
    #     plt.legend()
    #     plt.xlabel('frame')
    #     plt.ylabel('energy (J)')
    #     plt.title(f'Energy of {data_entry["data_name"]}')
    #     DATA_FOLDER = './imgs'
    #     # plt.savefig(f'{DATA_FOLDER}/{data_entry["data_name"]}-energy.png')
    #     plt.close()

    #     # 绘制动能
    #     plt.figure()
    #     plt.plot(data_entry['translational_energy'], label='translational energy')
    #     plt.plot(data_entry['rotational_energy'], label='rotational energy')
    #     plt.legend()
    #     plt.xlabel('frame')
    #     plt.ylabel('energy (J)')
    #     plt.title(f'Energy of {data_entry["data_name"]}')
    #     DATA_FOLDER = './imgs'
    #     # plt.savefig(f'{DATA_FOLDER}/{data_entry["data_name"]}-kinetic-energy.png')
    #     plt.close()

    #     # 绘制角速度
    #     plt.figure()
    #     plt.figure(figsize=(20, 10))
    #     for i in range(1, 22):
    #         plt.plot(data_entry['filtered_angular_velocity'][:, i], label=f'angular velocity: {i}')
    #     plt.legend()
    #     plt.xlabel('frame')
    #     plt.ylabel('-1')
    #     plt.title(f'Angular velocity of {data_entry["data_name"]}')
    #     DATA_FOLDER = './imgs'
    #     plt.savefig(f'{DATA_FOLDER}/{data_entry["data_name"]}-angular-velocity.png')
    #     plt.close()

    # # 绘制滤波后的机械能
    # plt.figure()
    # for data_entry in data:
    #     plt.plot(data_entry['filtered_mechanical_energy'], label=f'Filtered mechanical energy of {data_entry["data_name"]}')
    # plt.legend()
    # plt.xlabel('frame')
    # plt.ylabel('energy (J)')
    # plt.title(f'Energy of {len(data)} data')
    # DATA_FOLDER = './imgs'
    # plt.savefig(f'{DATA_FOLDER}/{len(data)}-filtered-mechanical-energy.png')
    # plt.close()

    # 绘制滤波后的动能
    plt.figure()
    for data_entry in data:
        plt.plot(data_entry['filtered_kinetic_energy'], label=f'Filtered kinetic energy of {data_entry["data_name"]}')  # 低通滤波后的动能
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('energy (J)')
    plt.title(f'Filtered kinetic energy of {len(data)} vanilla data')
    plt.savefig(f'./imgs/{len(data)}-filtered-kinetic_energy-{data_type}.png')
    plt.close()


if __name__ == '__main__':
    DATA_NAME = ['seat_1-frame_num_150', 'seat_5-frame_num_150']
    # DATA_NAME = ['seat_5-frame_num_150']
    output_1 = []
    output_2 = []
    for data_name in DATA_NAME:
        DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{data_name}'
        human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
        energy_entery = mechanical_energy_vanilla(human_params, FPS=30, data_name=data_name)
        output_1.append(energy_entery)
        energy_entery = mechanical_energy_mujoco(data_name=data_name)
        output_2.append(energy_entery)
    plot_energy(output_1, 'vanilla')
    plot_energy(output_2, 'mujoco')
