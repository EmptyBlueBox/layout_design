'''
Input: motion data
Output: mechanical energy consumption
'''
import rerun as rr
import pickle
import smplx
import torch
import numpy as np

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

MASS_DISTRIBUTION = np.array([Pelvis, Thigh/2, Thigh/2, Abdomen/2, Leg/2, Leg/2, Abdomen/2, Foot/4, Foot/4, Thorax/4,
                             Foot/4, Foot/4, Thorax/4, Thorax/4, Thorax/4, Hand, Upper_Arm/2, Upper_Arm/2, Forearm/2,
                             Forearm/2, Hand/2, Hand/2])

FPS = 30


def mechanical_energy(human_params):
    """
    The function `mechanical_energy` calculates the average work done by a human model based on their
    poses, orientation, and translation over a series of frames.

    Arguments:

    * `human_params`: The `human_params` dictionary contains the following keys:

    Returns:

    The function `mechanical_energy` returns the average work done by the human model per frame over all
    frames based on the provided human parameters.
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

    # 计算速度
    velocity = []
    for i in range(FRAME_NUM):
        if i == 0:
            velocity.append(np.zeros((22, 3)))
        else:
            velocity.append((key_points[i] - key_points[i-1]) * FPS)
    # 计算每一帧的机械能
    mechanical_energy = []
    for i in range(FRAME_NUM):
        key_points_i = key_points[i]
        # 计算每一帧的机械能
        energy = np.sum(MASS_DISTRIBUTION * key_points_i[:, 1] * 9.81)  # 乘以y轴坐标
        energy += np.sum(MASS_DISTRIBUTION * np.linalg.norm(velocity[i], axis=1) ** 2) / 2
        mechanical_energy.append(energy)
    # 计算总做功下界
    total_work = 0.
    for i in range(1, FRAME_NUM):
        if mechanical_energy[i] > mechanical_energy[i-1]:
            total_work += mechanical_energy[i] - mechanical_energy[i-1]  # 机械能增加的部分
    return total_work/FRAME_NUM  # 平均每帧做的功


if __name__ == '__main__':
    DATA_NAME = 'seat_1-frame_num_100'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
    DATA_NAME = 'seat_5-frame_num_100'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
    DATA_NAME = 'seat_1-frame_num_150'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
    DATA_NAME = 'seat_5-frame_num_150'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')

    DATA_NAME = 'seat_1-frame_num_50'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
    DATA_NAME = 'seat_5-frame_num_50'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
