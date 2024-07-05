# 获取当前脚本的目录
import sys
import smplx
import argparse
import os
import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R
from rerun.datatypes import Angle, RotationAxisAngle, Quaternion
import rerun as rr
import trimesh
import numpy
import utils.rotation_utils as ru


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


device = setup_device()
max_frame_num = 3  # 实际的帧数
frames_per_second = 1  # fps
selected_frame_num = np.linspace(0, int(max_frame_num/frames_per_second*120), max_frame_num, dtype=int)

human_list = [0, 1, 2, 3]  # ID
# object_list = ['bigsofa', 'smallsofa']
# object_list = ['bigsofa']
object_list = []


def compute_vertex_normals(vertices, faces):
    """
    使用向量化操作计算顶点法向量。
    """
    # 获取三角形的顶点
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 计算每个面的法向量
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # 将法向量累加到顶点上
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], normals)

    # 归一化顶点法向量
    norms = np.linalg.norm(vertex_normals, axis=1)
    vertex_normals = (vertex_normals.T / norms).T

    return vertex_normals


def load_human_mesh(id=0):
    human_model = smplx.create('/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=max_frame_num)

    data_root = '../dataset/livingroom_data05/smpl'
    data_list = os.listdir(data_root)
    data_list.sort()
    smpl_params = {
        'shapes': [],
        'poses': [],
        'orientation': [],
        'translation': [],
        'id': id,
    }

    for name in data_list:
        if name.endswith('.json'):
            name_int = int(name.split('.')[0])
            if name_int not in selected_frame_num:
                continue
            with open(os.path.join(data_root, name), 'r') as f:
                data = json.load(f)[id]
            smpl_params['shapes'].append(data['shapes'])
            smpl_params['poses'].append(data['poses'])
            smpl_params['orientation'].append(data['Rh'])
            smpl_params['translation'].append(data['Th'])
    smpl_params = {key: np.array(value) if isinstance(value, list) else value for key, value in smpl_params.items()}

    # 后续处理
    smpl_params['shapes'] = torch.tensor(smpl_params['shapes'].reshape(-1, 10), dtype=torch.float32)
    smpl_params['poses'] = torch.tensor(smpl_params['poses'].reshape(-1, 23, 3)[:, :-2], dtype=torch.float32)
    smpl_params['orientation'] = torch.tensor(smpl_params['orientation'].reshape(-1, 3), dtype=torch.float32)
    smpl_params['translation'] = torch.tensor(smpl_params['translation'].reshape(-1, 3), dtype=torch.float32)

    print(f'smpl_params.type:{smpl_params["shapes"].dtype}')
    print(f'smpl_params.shape:{smpl_params["shapes"].shape}')
    print(f'smpl_params.pose:{smpl_params["poses"].shape}')
    print(f'smpl_params.orientation:{smpl_params["orientation"].shape}')
    print(f'smpl_params.translation:{smpl_params["translation"].shape}')

    output = human_model(betas=smpl_params['shapes'],
                         body_pose=smpl_params['poses'],
                         global_orient=smpl_params['orientation'],
                         transl=smpl_params['translation'])
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    # 计算顶点法向量
    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)

    print(f'vertices.shape:{vertices.shape}')
    print(f'body_model.faces.shape:{human_model.faces.shape}')
    print(f'vertex_normals.shape:{vertex_normals.shape}')

    # human_mesh = {'vertices': vertices, 'faces': faces}
    human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

    return human_mesh


def load_object_params(object_name):
    data_root = f'../dataset/livingroom_data05/object/{object_name}/json'
    data_list = os.listdir(data_root)
    data_list.sort()

    object_params = {
        'translation': [],
        'orientation': [],
    }

    for name in data_list:
        if name.endswith('.json'):
            with open(os.path.join(data_root, name), 'r') as f:
                data = json.load(f)
            object_params['translation'].append(data['object_T'])
            object_params['orientation'].append(data['object_R'])
        else:
            data_list.remove(name)
    object_params = {key: np.array(value) if isinstance(value, list) else value for key, value in object_params.items()}

    object_params['translation'] = object_params['translation'].reshape(-1, 3)
    object_params['orientation'] = ru.rot6d_to_matrix(object_params['orientation'])    # 从前两列3*2算出3*3的旋转矩阵

    repeat_first_frame_num = int(data_list[0].split('.')[0])
    if (repeat_first_frame_num != 0):
        object_params['translation'] = np.concatenate(
            (np.tile(object_params['translation'][0], (repeat_first_frame_num-1, 1)), object_params['translation']), axis=0)
        object_params['orientation'] = np.concatenate(
            (np.tile(object_params['orientation'][0], (repeat_first_frame_num-1, 1, 1)), object_params['orientation']), axis=0)

    object_params['translation'] = object_params['translation'][selected_frame_num]
    object_params['orientation'] = object_params['orientation'][selected_frame_num]

    print(f'object_params[\'translation\'].shape:{object_params["translation"].shape}')
    print(f'object_params[\'orientation\'].shape:{object_params["orientation"].shape}')

    # 错误保存为转置, 要改正
    object_params['orientation'] = object_params['orientation'].transpose(0, 2, 1)
    # for i in range(object_params['orientation'].shape[0]):
    #     print(i, object_params['translation'][i])
    #     print(i, R.from_matrix(object_params['orientation'][i]).as_quat())

    return object_params


def load_human():
    human = []
    for id in human_list:
        human_mesh = load_human_mesh(id)  # (T, 6890, 3), (13776, 3)
        human.append(human_mesh)
    return human


def load_object():
    object = []
    for object_name in object_list:
        object_params = load_object_params(object_name)
        object.append(object_params)
    return object


def write_rerun(human: list, object: list):
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "rerun_example_dna_abacus")
    rr.set_time_seconds("stable_time", 0)

    for i in range(max_frame_num):
        time = i / frames_per_second
        rr.set_time_seconds("stable_time", time)

        for j in range(len(human_list)):
            rr.log(
                f'human/human_{j}',
                rr.Mesh3D(
                    vertex_positions=human[j]['vertices'][i],
                    triangle_indices=human[j]['faces'],
                    vertex_normals=human[j]['vertex_normals'][i],
                ),
            )

        for j in range(len(object_list)):
            rr_transform = rr.Transform3D(
                rotation=Quaternion(xyzw=R.from_matrix(object[j]['orientation'][i]).as_quat()),
                translation=object[j]['translation'][i]
            )
            object_obj = f'../dataset/object_obj/{object_list[j]}/{object_list[j]}_face1000.obj'
            rr.log(f'object/{object_list[j]}', rr_transform)
            rr.log(f'object/{object_list[j]}', rr.Asset3D(path=object_obj))

    rr.script_teardown(args)


def main():
    human = load_human()  # 和 human_list 一一对应
    object = load_object()  # 和 object_list 一一对应
    write_rerun(human, object)


if __name__ == '__main__':
    main()
