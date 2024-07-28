import argparse
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
from utils.visualization_utils import write_object_human
import rerun as rr


def load_data(step=50):
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
        # data[tuple(object_params.keys())] = [data_entry]  # 只取第一个数据

    for key in data.keys():
        for i in range(len(data[key])):
            for single_obj_name in key:
                data[key][i]['object_params'][single_obj_name]['location'] = data[key][i]['object_params'][single_obj_name]['location'][::step]
                data[key][i]['object_params'][single_obj_name]['rotation'] = data[key][i]['object_params'][single_obj_name]['rotation'][::step]
            data[key][i]['human_params']['translation'] = data[key][i]['human_params']['translation'][::step]
            data[key][i]['human_params']['orientation'] = data[key][i]['human_params']['orientation'][::step]
            data[key][i]['human_params']['poses'] = data[key][i]['human_params']['poses'][::step]
            # 人体和物体的 x z 位置归零
            base_obj_name = key[0]
            for single_obj_name in key:
                if single_obj_name != base_obj_name:
                    data[key][i]['object_params'][single_obj_name]['location'][:, [0, 2]
                                                                               ] -= data[key][i]['object_params'][base_obj_name]['location'][0, [0, 2]]
            data[key][i]['human_params']['translation'][:, [0, 2]] -= data[key][i]['object_params'][base_obj_name]['location'][0, [0, 2]]
            data[key][i]['object_params'][base_obj_name]['location'][:, [0, 2]] -= data[key][i]['object_params'][base_obj_name]['location'][0, [0, 2]]
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


class visualize_data:
    def __init__(self, data):
        self.data = data

        self.object_num = len(data)
        self.obj_name = list(data.keys())

    def __call__(self, x):
        '''
        x: (K, 3) 二维数组, K 为物体数量, 3 为每个物体的xz位移和y旋转
        '''
        max_frame = 0
        for key in self.data.keys():
            for i in range(len(self.data[key])):
                length = self.data[key][i]['human_params']['translation'].shape[0]
                length = self.data[key][i]['human_params']['orientation'].shape[0]
                length = self.data[key][i]['human_params']['poses'].shape[0]
                for single_obj_name in key:
                    length = self.data[key][i]['object_params'][single_obj_name]['location'].shape[0]
                    length = self.data[key][i]['object_params'][single_obj_name]['rotation'].shape[0]
                max_frame = max(max_frame, length)
        # print(f'max_frame: {max_frame}')

        print('writing rerun...')
        parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
        rr.script_add_args(parser)
        args = parser.parse_args()
        rr.script_setup(args, 'CMA-ES')
        rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
        rr.set_time_seconds("stable_time", 0)

        # 画墙
        rr.log(
            "wall",
            rr.Boxes3D(
                centers=[config.ROOM_SHAPE_X/2, .8, config.ROOM_SHAPE_Z/2],
                half_sizes=[config.ROOM_SHAPE_X/2, .8, config.ROOM_SHAPE_Z/2],
                radii=0.01,
                colors=(0, 0, 255),
                labels="blue",
            ),
        )
        query_point = {}
        for i in range(self.object_num):
            query_point[self.obj_name[i]] = []
            for j in range(len(self.data[self.obj_name[i]])):
                # print(f'object: {self.obj_name[i]}, x: {x[i]}')
                human_params = self.data[self.obj_name[i]][j]['human_params']
                object_params = self.data[self.obj_name[i]][j]['object_params']
                query = write_object_human(object_params,
                                           human_params,
                                           np.array([x[i, 0], 0, x[i, 1]]),
                                           x[i, 2],
                                           max_frame_num=max_frame)
                query_point[self.obj_name[i]].append(query)

        rr.script_teardown(args)
        print('write rerun done!\n')

        return query_point


visualizer = None


class objective_function:
    def __init__(self, data, step=10, obj_loss_scale=5, object_point=1000, human_point=2000, oversample_factor=2):
        self.data = data
        self.step = step
        self.obj_loss_scale = obj_loss_scale
        self.object_point = object_point
        self.human_point = human_point
        self.oversample_factor = oversample_factor

        self.object_num = len(data)
        self.obj_name = list(data.keys())  # [(base, move), ()]
        self.query_point = None

    # 计算物体人点云和另一个物体之间的碰撞
    def loss_motion_obj(self, obj_name1, obj_name2, info1, info2):
        obj_num_1 = len(obj_name1)  # 一个物体组中的物体数量
        obj_num_2 = len(obj_name2)
        motion_num1 = len(self.data[obj_name1])  # 一个物体组中的运动轨迹数量
        motion_num2 = len(self.data[obj_name2])

        translation_1 = np.array([info1[0], 0, info1[1]])
        orientation_1 = R.from_rotvec(np.array([0, info1[2], 0]))
        translation_2 = np.array([info2[0], 0, info2[1]])
        orientation_2 = R.from_rotvec(np.array([0, info2[2], 0]))

        query_point_1 = self.query_point[obj_name1]
        query_point_2 = self.query_point[obj_name2]

        # 计算两个物体之间的碰撞
        motion_clip_penetration = []  # 每个元素是一个 motion clip 和另一个物体的碰撞深度总和, 最后只需要取最大值即可
        # print(f'loop1: {motion_num1}, loop2: {motion_num2}')
        for i in range(motion_num1):  # 对第一个物体组的每一个 motion trajectory 是一个点云
            for j in range(motion_num2):  # 对第二个物体组的每一个 motion trajectory 是一个点云
                point_cloud_1_when_2_normal = orientation_2.inv().apply(query_point_1[i]-translation_2)+translation_2
                # print(f'obj_name2 length: {len(obj_name2)}, j//len(obj_name2): {j//motion_num1}')  # 测试
                signed_distence = query_sdf_normalized(point_cloud_1_when_2_normal, obj_name2[0])  # 一个点云和一个物体的碰撞深度
                # print(f'signed_distence min: {np.min(signed_distence)}')
                negative_mask = signed_distence < 0
                loss_1 = -np.sum(signed_distence[negative_mask])
                # print(f'loss_1: {loss_1}')

                point_cloud_2_when_1_normal = orientation_1.inv().apply(query_point_2[j]-translation_1)+translation_1
                signed_distence = query_sdf_normalized(point_cloud_2_when_1_normal, obj_name1[0])
                negative_mask = signed_distence < 0
                loss_2 = -np.sum(signed_distence[negative_mask])
                # print(f'loss_2: {loss_2}')

                motion_clip_penetration.append(loss_1+loss_2)

        motion_clip_penetration = np.array(motion_clip_penetration)
        return np.max(motion_clip_penetration)

    def loss_motion_wall(self, motion_name, info):
        query_point = self.query_point[motion_name]
        motion_num = len(self.data[motion_name])  # 一个物体组中的运动轨迹数量
        translation_1 = np.array([info[0], 0, info[1]])

        # 计算人体和墙之间的碰撞
        x_min = 0
        x_max = config.ROOM_SHAPE_X
        z_min = 0
        z_max = config.ROOM_SHAPE_Z

        motion_clip_penetration = []  # 每个元素是一个 motion clip 和墙的碰撞深度总和, 最后只需要取最大值即可
        for i in range(motion_num):  # 对第一个物体组的每一个 motion trajectory 是一个点云
            point_cloud = query_point[i]
            signed_distence_x = np.maximum(point_cloud[:, 0]-x_max, x_min-point_cloud[:, 0])
            signed_distence_z = np.maximum(point_cloud[:, 2]-z_max, z_min-point_cloud[:, 2])
            signed_distence = np.maximum(signed_distence_x, signed_distence_z)
            penetration_mask = signed_distence > 0

            weight = np.exp(-np.linalg.norm(point_cloud[penetration_mask]-translation_1, axis=1))
            loss = np.sum(signed_distence[penetration_mask]*weight)
            motion_clip_penetration.append(loss)

        motion_clip_penetration = np.array(motion_clip_penetration)
        return np.max(motion_clip_penetration)

    def __call__(self, x):
        x = x.reshape(self.object_num, 3)
        # print(f'objective_function x: {x}')
        self.query_point = visualizer(x)

        # 计算所有物体之间的碰撞
        print('Calculating loss_motion_obj...')
        loss_obj2obj = 0
        for i in range(self.object_num):
            for j in range(i+1, self.object_num):
                loss_obj2obj += self.loss_motion_obj(self.obj_name[i], self.obj_name[j], x[i], x[j])

        # 计算所有动作和墙之间的碰撞
        print('Calculating loss_motion_wall...')
        loss_human_obj = 0
        for i in range(self.object_num):
            loss_human_obj += self.loss_motion_wall(self.obj_name[i], x[i])

        print(f'loss_motion_obj: {loss_obj2obj}, loss_motion_wall: {loss_human_obj}')
        return loss_obj2obj * self.obj_loss_scale + loss_human_obj


def constraints(x):
    xmin = 0
    xmax = config.ROOM_SHAPE_X
    zmin = 0
    zmax = config.ROOM_SHAPE_Z
    output = []
    x = x.reshape(-1, 3)
    # print(f'x shape: {x.shape}')
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
    print(f'Initial translation: {translation_init}, orientation: {orientation_init}')
    x0 = np.concatenate((translation_init, orientation_init), axis=1).reshape(-1)
    sigma0 = 1

    obj = objective_function(data, step=100, object_point=3000)
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
    visualizer = visualize_data(data)
    output = optimize(data)
    print(output)
