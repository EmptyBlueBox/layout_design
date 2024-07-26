import numpy as np
import cma
import os
import regex as re
import pickle
import config
from scipy.spatial.transform import Rotation as R
from utils.mesh_utils import query_sdf_normalized, sample_point_cloud_from_mesh, farthest_point_sampling
import trimesh
import smplx
import torch
from tqdm import tqdm


def load_data():
    DATA_FOLDER = '/Users/emptyblue/Documents/Research/layout_design/dataset/data_pair-hand_picked'
    subdata_list = os.listdir(DATA_FOLDER)
    subdata_list.sort()
    subdata_list = [name for name in subdata_list if re.match(r'seg_\d+-obj_\w+-end_\d+\.\d+-len_\d+', name)]  # seg_3-obj_fridge-end_0.382-len_130

    data = {}
    for subdata_name in subdata_list:
        subdata_object_path = os.path.join(DATA_FOLDER, subdata_name, 'object-params.pkl')
        object_params = pickle.load(open(subdata_object_path, 'rb'))
        data[tuple(object_params.keys())] = []

    for subdata_name in subdata_list:
        subdata_human_path = os.path.join(DATA_FOLDER, subdata_name, 'human-params.pkl')
        subdata_object_path = os.path.join(DATA_FOLDER, subdata_name, 'object-params.pkl')
        human_params = pickle.load(open(subdata_human_path, 'rb'))
        object_params = pickle.load(open(subdata_object_path, 'rb'))
        data_entry = {'human_params': human_params, 'object_params': object_params}
        data[tuple(object_params.keys())].append(data_entry)

    return data


def print_data(data: dict):
    print(f'All objects: {data.keys()}')
    for key in data.keys():
        print(f'Object {key} has {len(data[key])} data entries')
    print(f"One data entry's human_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['human_params'].keys()}")
    print(f"One data entry's object_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['object_params'].keys()}")
    # Print trajectory scale
    for key in data.keys():
        trajectory = data[key][0]['human_params']['translation']
        delta_x = trajectory[0][0]-trajectory[-1][0]
        delta_y = trajectory[0][1]-trajectory[-1][1]
        delta_z = trajectory[0][2]-trajectory[-1][2]
        scale = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        print(f'Object {key}, trajectory scale: {scale}, delta_x: {delta_x}, delta_y: {delta_y}, delta_z: {delta_z}')
    print(f"One data entry\'s object_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['object_params']['cabinet_base_01'].keys()}\n")


class objective_function:
    def __init__(self, data, step=10, obj_loss_scale=5, object_point=1000, human_point=2000, oversample_factor=2):
        self.data = data
        self.step = step
        self.obj_loss_scale = obj_loss_scale
        self.object_point = object_point
        self.human_point = human_point
        self.oversample_factor = oversample_factor

        self.object_num = len(data)
        self.obj_name = list(data.keys())

    def loss_2obj(self, obj_name1, obj_name2, info1, info2):
        obj_num_1 = len(obj_name1)  # 一个物体组中的物体数量
        obj_num_2 = len(obj_name2)
        motion_num1 = len(self.data[obj_name1])  # 一个物体组中的运动轨迹数量
        motion_num2 = len(self.data[obj_name2])

        # 两个物体(组)之间的碰撞
        translation_1 = np.array([info1[0], 0, info1[1]])
        orientation_1 = R.from_rotvec(np.array([0, info1[2], 0]))
        translation_2 = np.array([info2[0], 0, info2[1]])
        orientation_2 = R.from_rotvec(np.array([0, info2[2], 0]))

        # 加载第一个物体的 query 点云
        point_clouds_1 = []  # 每一个元素是一个 (T, K, 3) 点云对应一个物品的一个 motion trajectory, 如果有多个物体则是两个 (T, K, 3) 点云
        original_translation_1 = []  # 每一个 motion trajectory 都是一个 (T, 3) 位移
        original_orientation_1 = []  # 每一个 motion trajectory 都是一个 (T, ) R 对象
        for single_obj_name in obj_name1:  # 对一组物品中的每一个物品
            obj_path = os.path.join(config.OBJECT_ORIGINAL_PATH, single_obj_name+'.obj')
            mesh = trimesh.load_mesh(obj_path)
            point_cloud = sample_point_cloud_from_mesh(mesh, self.object_point, self.oversample_factor)
            for original_params in self.data[obj_name1]:  # 对每一个 motion trajectory, 采集了没有运动的点云, 还需要 1.把点云放到路径上面 2.进行 x 的旋转和平移
                # print(f'111:{original_params.keys()}')
                original_translation = original_params['object_params'][single_obj_name]['location'][::self.step]
                original_translation_1.append(original_translation)
                original_orientation = R.from_euler('xyz', original_params['object_params'][single_obj_name]['rotation'][::self.step])  # T 个 R 对象
                original_orientation_1.append(original_orientation)
                total_translation = orientation_1.apply(original_translation)+translation_1  # T 个 3D 向量
                total_orientation = orientation_1*original_orientation  # T 个 R 对象
                point_cloud_all_frame = np.zeros(
                    (original_translation.shape[0], point_cloud.shape[0], point_cloud.shape[1]))  # (T, K, 3), 每个 R 对象都作用在所有点上
                for i in range(original_translation.shape[0]):
                    point_cloud_all_frame[i] = total_orientation[i].apply(point_cloud)+total_translation[i]
                point_clouds_1.append(point_cloud_all_frame)

        # 加载第二个物体的 query 点云, 所有点云按照时间排序排成 (T, K, 3) 的矩阵
        point_clouds_2 = []
        original_translation_2 = []
        original_orientation_2 = []
        for single_obj_name in obj_name2:
            obj_path = os.path.join(config.OBJECT_ORIGINAL_PATH, single_obj_name+'.obj')
            mesh = trimesh.load_mesh(obj_path)
            point_cloud = sample_point_cloud_from_mesh(mesh, self.object_point, self.oversample_factor)
            for original_params in self.data[obj_name2]:
                original_translation = original_params['object_params'][single_obj_name]['location'][::self.step]
                original_translation_2.append(original_translation)
                original_orientation = R.from_euler('xyz', original_params['object_params'][single_obj_name]['rotation'][::self.step])
                original_orientation_2.append(original_orientation)
                total_translation = orientation_2.apply(original_translation)+translation_2
                total_orientation = orientation_2*original_orientation
                point_cloud_all_frame = np.zeros((original_translation.shape[0], point_cloud.shape[0], point_cloud.shape[1]))
                for i in range(original_translation.shape[0]):
                    point_cloud_all_frame[i] = total_orientation[i].apply(point_cloud)+total_translation[i]
                point_clouds_2.append(point_cloud_all_frame)

        # 计算两个物体之间的碰撞
        frame_penetration = []  # 每个元素是一帧的碰撞深度总和, 最后只需要取最大值即可
        print(f'loop1: {len(point_clouds_1)}, loop2: {len(point_clouds_2)}, loop3: {point_clouds_1[0].shape[0]}, loop4: {point_clouds_2[0].shape[0]}')
        for i in range(len(point_clouds_1)):  # 对第一个物体组的每一个 motion trajectory
            for j in range(len(point_clouds_2)):  # 对第二个物体组的每一个 motion trajectory
                for k in range(point_clouds_1[i].shape[0]):  # 对第一个物体的第 i 个 motion trajectory 的每一帧
                    for ll in range(point_clouds_2[j].shape[0]):  # 对第二个物体的第 j 个 motion trajectory 的每一帧
                        # print(f'i: {i}, j: {j}, k: {k}, ll: {ll}')
                        new_point_cloud_1 = original_orientation_2[j][ll].inv().apply(point_clouds_1[i][k]-original_translation_2[j][ll])
                        # print(f'obj_name2 length: {len(obj_name2)}, j//len(obj_name2): {j//motion_num1}')  # 测试
                        signed_distence = query_sdf_normalized(new_point_cloud_1, obj_name2[j//motion_num2])  # 一个点云和一个物体的碰撞深度
                        negative_mask = signed_distence < 0
                        loss_1 = -np.sum(signed_distence[negative_mask])

                        new_point_cloud_2 = original_orientation_1[i][k].inv().apply(point_clouds_2[j][ll]-original_translation_1[i][k])
                        signed_distence = query_sdf_normalized(new_point_cloud_2, obj_name1[i//motion_num1])
                        negative_mask = signed_distence < 0
                        loss_2 = -np.sum(signed_distence[negative_mask])

                        frame_penetration.append(loss_1+loss_2)
        frame_penetration = np.array(frame_penetration)

        return np.max(frame_penetration)

    def loss_human_obj(self, human_name, obj_name, info1, info2):
        human_num = len(human_name)  # 人组中的物体数量
        human_motion_num = len(self.data[human_name])  # 人组中的运动轨迹数量
        obj_num = len(obj_name)  # 物体组中的物体数量
        obj_motion_num = len(self.data[obj_name])  # 物体组中的运动轨迹数量

        # 一个物体(组)与所有人之间的碰撞
        translation_1 = np.array([info1[0], 0, info1[1]])
        orientation_1 = R.from_rotvec(np.array([0, info1[2], 0]))
        translation_2 = np.array([info2[0], 0, info2[1]])
        orientation_2 = R.from_rotvec(np.array([0, info2[2], 0]))

        # 加载人的 query 点云
        point_clouds_human = []
        for human_params in self.data[human_name]:
            original_translation = human_params['human_params']['translation'][::self.step]
            original_orientation = human_params['human_params']['orientation'][::self.step]
            original_orientation = (orientation_1*R.from_rotvec(original_orientation)).as_rotvec()
            original_poses = human_params['human_params']['poses'][::self.step]
            frame_num = original_translation.shape[0]
            human_model = smplx.create(model_path=config.SMPL_MODEL_PATH,
                                       model_type='smplx',
                                       gender='neutral',
                                       use_face_contour=False,
                                       num_betas=10,
                                       num_expression_coeffs=10,
                                       ext='npz',
                                       batch_size=frame_num)
            output = human_model(body_pose=torch.tensor(original_poses, dtype=torch.float32),
                                 global_orient=torch.tensor(original_orientation, dtype=torch.float32),
                                 transl=torch.tensor(original_translation, dtype=torch.float32)
                                 )
            point_cloud = output.vertices.detach().cpu().numpy()
            point_cloud = farthest_point_sampling(point_cloud, self.human_point)

            first_key = next(iter(human_params['object_params']))
            object_translation = human_params['object_params'][first_key]['location'][::self.step][:, None, :]
            # print(f'point_cloud shape: {point_cloud.shape}, object_translation shape: {object_translation.shape}, object_translation: {object_translation.shape}, translation_1 shape: {translation_1.shape}')
            point_cloud_all_frame = np.zeros_like(point_cloud)
            for i in range(frame_num):
                point_cloud_all_frame[i] = (orientation_1.apply(point_cloud[i]-object_translation[i])) + object_translation[i] + translation_1
            # print(f'len: {len(point_cloud_all_frame)}, point_cloud_all_frame shape: {point_cloud_all_frame.shape}')
            if len(point_cloud_all_frame.shape) == 2:
                point_cloud_all_frame = point_cloud_all_frame[None, ...]  # 扩展帧数维度
            assert len(point_cloud_all_frame.shape) == 3
            point_clouds_human.append(point_cloud_all_frame)

        # 加载物体的 query 点云
        point_clouds_obj = []
        original_translation_obj = []
        original_orientation_obj = []
        for single_obj_name in obj_name:
            obj_path = os.path.join(config.OBJECT_ORIGINAL_PATH, single_obj_name+'.obj')
            mesh = trimesh.load_mesh(obj_path)
            point_cloud = sample_point_cloud_from_mesh(mesh, self.object_point, self.oversample_factor)
            for original_params in self.data[obj_name]:
                original_translation = original_params['object_params'][single_obj_name]['location'][::self.step]
                original_translation_obj.append(original_translation)
                original_orientation = R.from_euler('xyz', original_params['object_params'][single_obj_name]['rotation'][::self.step])
                original_orientation_obj.append(original_orientation)
                total_translation = orientation_2.apply(original_translation)+translation_2
                total_orientation = orientation_2*original_orientation
                point_cloud_all_frame = np.zeros((original_translation.shape[0], point_cloud.shape[0], point_cloud.shape[1]))
                for i in range(original_translation.shape[0]):
                    point_cloud_all_frame[i] = total_orientation[i].apply(point_cloud)+total_translation[i]
                assert len(point_cloud_all_frame.shape) == 3
                point_clouds_obj.append(point_cloud_all_frame)

        # 计算人和物体之间的碰撞
        frame_penetration = []
        print(
            f'loop1: {len(point_clouds_human)}, loop2: {len(point_clouds_obj)}, loop3: {point_clouds_human[0].shape[0]}, loop4: {point_clouds_obj[0].shape[0]}')
        for i in range(len(point_clouds_human)):
            for j in range(len(point_clouds_obj)):
                for k in range(point_clouds_human[i].shape[0]):
                    for ll in range(point_clouds_obj[j].shape[0]):
                        new_point_cloud_human = original_orientation_obj[j][ll].inv().apply(point_clouds_human[i][k]-original_translation_obj[j][ll])
                        # print(f'new_point_cloud_human shape: {new_point_cloud_human.shape}')
                        signed_distence = query_sdf_normalized(new_point_cloud_human, obj_name[j//obj_motion_num])
                        negative_mask = signed_distence < 0
                        loss_1 = -np.sum(signed_distence[negative_mask])
                        frame_penetration.append(loss_1)
        frame_penetration = np.array(frame_penetration)

        return np.max(frame_penetration)

    def __call__(self, x):
        x = x.reshape(self.object_num, 3)

        # 计算所有物体之间的碰撞
        print('Calculating loss_obj2obj...')
        loss_obj2obj = 0
        for i in range(self.object_num):
            for j in range(i+1, self.object_num):
                loss_obj2obj += self.loss_2obj(self.obj_name[i], self.obj_name[j], x[i], x[j])

        # 计算所有人与物体之间的碰撞
        print('Calculating loss_human_obj...')
        loss_human_obj = 0
        for i in range(self.object_num):
            for j in range(i+1, self.object_num):
                loss_human_obj += self.loss_human_obj(self.obj_name[i], self.obj_name[j], x[i], x[j])

        print(f'loss_obj2obj: {loss_obj2obj}, loss_human_obj: {loss_human_obj}')
        return loss_obj2obj * self.obj_loss_scale + loss_human_obj


def constraints(x):
    xmin = 0
    xmax = config.ROOM_SHAPE_X
    zmin = 0
    zmax = config.ROOM_SHAPE_Z
    output = []
    x = x.reshape(-1, 3)
    print(f'x shape: {x.shape}')
    for i in range(x.shape[0]):
        output.append(- x[i][0] + xmin)  # x >= xmin
        output.append(x[i][0] - xmax)  # x <= xmax
        output.append(- x[i][1] + zmin)  # z >= zmin
        output.append(x[i][1] - zmax)   # z <= zmax
        output.append(- x[i][2] - np.pi)  # orientation >= -pi
        output.append(x[i][2] - np.pi)   # orientation <= pi

    return output


def optimize(data: dict):
    object_num = len(data)
    translation_init = np.random.rand(object_num, 2)*np.array([config.ROOM_SHAPE_X, config.ROOM_SHAPE_Z])
    orientation_init = np.random.rand(object_num, 1)*2*np.pi-np.pi
    print(f'Initial translation: {translation_init.shape}, orientation: {orientation_init.shape}')
    x0 = np.concatenate((translation_init, orientation_init), axis=1).reshape(-1)
    sigma0 = 1

    obj = objective_function(data, step=100)
    cfun = cma.ConstrainedFitnessAL(obj, constraints)  # unconstrained function with adaptive Lagrange multipliers

    x, es = cma.fmin2(cfun, x0, sigma0, {'tolstagnation': 0}, callback=cfun.update)
    x = es.result.xfavorite  # the original x-value may be meaningless
    constraints(x)  # show constraint violation values

    output = {
        'translation': x[:, :2],
        'orientation': x[:, 2]
    }
    return output


if __name__ == '__main__':
    data = load_data()
    print_data(data)
    output = optimize(data)
    print(output)
