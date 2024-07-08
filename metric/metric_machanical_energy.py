import pickle
import smplx
import torch
import numpy as np
import matplotlib.pyplot as plt

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


def mechanical_energy(human_params, FPS=30, data_name=None):
    """
    The function `mechanical_energy` calculates and visualizes mechanical energy and power of a human
    model based on input parameters.

    Arguments:

    * `human_params`: The `human_params` parameter seems to contain information about a human model,
    including poses, orientation, and translation. It is used to create a human model using the `smplx`
    library and then calculate mechanical energy and power based on the model's movements.
    * `FPS`: The `FPS` parameter in the `mechanical_energy` function stands for Frames Per Second, which
    represents the number of frames captured or processed per second in a video or animation sequence.
    It is used to calculate the velocity of the human model's key points based on the poses provided in
    the `
    * `data_name`: The `data_name` parameter is a string that represents the name of the data being used
    in the function. It is used to save the generated plot with a specific name in a designated folder.
    If `data_name` is provided, the plot will be saved in a folder specific to that data.

    Returns:

    The function `mechanical_energy` returns the array `power`, which represents the lower bound of the
    power exerted by the human body in each frame of the animation.
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
    power = np.zeros_like(mechanical_energy)  # 人体做功功率下界
    for i in range(1, FRAME_NUM):
        if mechanical_energy[i] > mechanical_energy[i-1]:
            power[i] = mechanical_energy[i] - mechanical_energy[i-1]  # 机械能增加的部分

    # 画图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左边的Y轴：机械能
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Mechanical Energy', color='blue')
    ax1.plot(range(FRAME_NUM), mechanical_energy, label='Mechanical Energy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 右边的Y轴：功率
    ax2 = ax1.twinx()  # 创建共用X轴的第二个Y轴
    ax2.set_ylabel('Power', color='red')
    ax2.plot(range(FRAME_NUM), power, label='Power LBO', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 添加标题
    plt.title('Mechanical Energy and Power')

    # 显示图例
    fig.tight_layout()  # 调整布局以防止标签重叠

    # 显示网格
    ax1.grid(True)

    # 显示图表
    if data_name is None:
        plt.savefig('./power.png')
    else:
        DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{data_name}'
        plt.savefig(f'{DATA_FOLDER}/power.png')

    return power


if __name__ == '__main__':
    DATA_NAME = 'seat_1-frame_num_150'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
    DATA_NAME = 'seat_5-frame_num_150'
    DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
    human_params = pickle.load(open(f'{DATA_FOLDER}/human-params.pkl', 'rb'))
    print(f'data {DATA_NAME} energy: {mechanical_energy(human_params)}')
