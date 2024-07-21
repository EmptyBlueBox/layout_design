import numpy as np
import cma
import os
import regex as re
import pickle

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
    print(f'All objects: {data.keys()}')
    for key in data.keys():
        print(f'Object {key} has {len(data[key])} data entries')
    print(f"One data entry's human_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['human_params'].keys()}")
    print(f"One data entry's object_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['object_params'].keys()}")
    return data


def optimize():
    pass


if __name__ == '__main__':
    data = load_data()
