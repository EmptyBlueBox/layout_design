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
import pickle


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


TRUMANS_PATH = '/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS'

device = setup_device()

seg_num = 1  # 可视化哪个seg

seg_begin_list = np.load(TRUMANS_PATH+'/seg_begin.npy')
seg_begin = seg_begin_list[seg_num]
seg_end = seg_begin_list[seg_num+1] if seg_num+1 < seg_begin_list.shape[0] else len(seg_begin_list)
seg_frame_num = seg_end - seg_begin

seg_name_list = np.load(TRUMANS_PATH+'/seg_name.npy')
seg_name = seg_name_list[seg_begin]

start_frame_ratio = 0.44  # 选择开始的帧数比例
end_frame_ratio = 0.5  # 选择停止的帧数比例
max_frame_num = min(200, int(seg_frame_num*(end_frame_ratio-start_frame_ratio)))  # 实际的帧数
print(f'max_frame_num: {max_frame_num}')
selected_frame = np.linspace(int(start_frame_ratio*seg_end+(1-start_frame_ratio)*seg_begin),
                             int((1-end_frame_ratio)*seg_end+end_frame_ratio*seg_begin),
                             max_frame_num,
                             dtype=int)
frames_per_second = max_frame_num / (seg_frame_num * (end_frame_ratio - start_frame_ratio)) * 60
print(f'frames_per_second: {frames_per_second}')


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


def load_human():

    human_model = smplx.create(model_path='/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=max_frame_num)

    smpl_params = {
        'poses': np.load(TRUMANS_PATH+'/human_pose.npy')[selected_frame],  # (max_frame_num, 63)
        'orientation': np.load(TRUMANS_PATH+'/human_orient.npy')[selected_frame],  # (max_frame_num, 3)
        'translation': np.load(TRUMANS_PATH+'/human_transl.npy')[selected_frame],  # (max_frame_num, 3)
    }

    # 后续处理
    smpl_params['poses'] = torch.tensor(smpl_params['poses'].reshape(-1, 21, 3), dtype=torch.float32)
    smpl_params['orientation'] = torch.tensor(smpl_params['orientation'].reshape(-1, 3), dtype=torch.float32)
    smpl_params['translation'] = torch.tensor(smpl_params['translation'].reshape(-1, 3), dtype=torch.float32)

    # print(f'smpl_params.pose: {smpl_params["poses"].shape}')
    # print(f'smpl_params.orientation: {smpl_params["orientation"].shape}')
    # print(f'smpl_params.translation: {smpl_params["translation"].shape}')

    output = human_model(body_pose=smpl_params['poses'],
                         global_orient=smpl_params['orientation'],
                         transl=smpl_params['translation'])
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    # 计算顶点法向量
    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)

    # print(f'vertices.shape: {vertices.shape}')
    # print(f'body_model.faces.shape: {human_model.faces.shape}')
    # print(f'vertex_normals.shape: {vertex_normals.shape}')

    human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

    return human_mesh


def load_object():
    object: dict = numpy.load(TRUMANS_PATH+f'/Object_all/Object_pose/{seg_name}.npy', allow_pickle=True).item()  # 这是一个被{}包起来的字典, 要做一下解包 .item()

    for key in object.keys():
        object[key]['rotation'] = np.array(object[key]['rotation'])[selected_frame-seg_begin]
        object[key]['location'] = np.array(object[key]['location'])[selected_frame-seg_begin]
        object_name = key
        object_obj_path = f'../dataset/TRUMANS/Object_all/Object_mesh/{object_name}.obj'
        object[key]['vertices'] = trimesh.load(object_obj_path).vertices
        object[key]['faces'] = trimesh.load(object_obj_path).faces
        object[key]['vertex_normals'] = compute_vertex_normals(object[key]['vertices'], object[key]['faces'])

    return object


def load_scene():
    scene_flag = np.load(TRUMANS_PATH+'/scene_flag.npy')[seg_begin]  # 当前seg的场景标志
    scene_list = np.load(TRUMANS_PATH+'/scene_list.npy')  # 一个包含所有场景的列表
    scene_name = scene_list[scene_flag]  # 一个场景的名字
    scene_path = f'../dataset/TRUMANS/Scene_mesh/{scene_name}.obj'
    scene = {
        'vertices': trimesh.load(scene_path).vertices,
        'faces': trimesh.load(scene_path).faces,
        'vertex_normals': compute_vertex_normals(trimesh.load(scene_path).vertices, trimesh.load(scene_path).faces)
    }

    return scene


def write_rerun(human: dict, object: dict, scene: dict = None):
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, f'TRUMANS seg: {seg_num}')
    rr.set_time_seconds("stable_time", 0)

    for i in range(max_frame_num):
        time = i / frames_per_second
        rr.set_time_seconds("stable_time", time)

        rr.log(
            'human',
            rr.Mesh3D(
                vertex_positions=human['vertices'][i],
                triangle_indices=human['faces'],
                vertex_normals=human['vertex_normals'][i],
            ),
        )

        for key in object.keys():  # 一个frame中遍历所有object:
            # 只要名字里有 chair 的
            if 'chair' not in key:
                continue

            rr_transform = rr.Transform3D(
                rotation=Quaternion(xyzw=R.from_euler('xyz', object[key]['rotation'][i]).as_quat()),
                translation=object[key]['location'][i],
            )

            rr.log(f'object/{key}', rr_transform)
            rr.log(f'object/{key}',
                   rr.Mesh3D(
                       vertex_positions=object[key]['vertices'],
                       triangle_indices=object[key]['faces'],
                       vertex_normals=object[key]['vertex_normals'],
                   ),
                   )

        if scene is not None:
            rr.log(
                'scene',
                rr.Mesh3D(
                    vertex_positions=scene['vertices'],
                    triangle_indices=scene['faces'],
                    vertex_normals=scene['vertex_normals'],
                ),
            )

    rr.script_teardown(args)


def main():
    human = load_human()
    object = load_object()
    scene = None
    # scene = load_scene()
    write_rerun(human=human, object=object, scene=scene)


if __name__ == '__main__':
    main()
