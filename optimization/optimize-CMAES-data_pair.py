import numpy as np
import cma
import os
import regex as re
import pickle
import config

# Load data


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


class objective_function:
    def __init__(self, data, step=10, obj_loss_scale=5):
        self.data = data
        self.step = step
        self.obj_loss_scale = obj_loss_scale
        self.object_num = len(data)

    def loss_2obj(self, obj_name1, obj_name2):
        # 两个物体(组)之间的碰撞
        pass

    def loss_human_obj(self, human_params, obj_name):
        # 一个物体(组)与所有人之间的碰撞
        pass

    def __call__(self, x):
        x = x.reshape(self.object_num, 3)
        translation = x[:, :2]
        orientation = x[:, 2]

        # 计算所有物体之间的碰撞
        for i in range(self.object_num):
            for j in range(i+1, self.object_num):
                self.loss_2obj(list(self.data.keys())[i], list(self.data.keys())[j])
        # 计算所有人与物体之间的碰撞


def optimize(data: dict):
    object_num = len(data)
    translation_init = np.random.rand(object_num, 2)*np.array([config.SHAPE_X, config.SHAPE_Z])
    orientation_init = np.random.rand(object_num)*2*np.pi-np.pi
    x0 = np.concatenate((translation_init, orientation_init), axis=1).reshape(-1)
    sigma0 = 1
    obj = objective_function(data)


if __name__ == '__main__':
    data = load_data()
    print_data(data)
