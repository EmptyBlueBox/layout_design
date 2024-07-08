import smplx
import argparse
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from rerun.datatypes import Quaternion
import rerun as rr
import trimesh
import pickle
from utils.mesh_utils import compute_vertex_normals

DATA_NAME = 'seat_5'
DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'

smpl_params = pickle.load(open(DATA_FOLDER+'/human-params.pkl', 'rb'))  # 读取 smpl_params
FRAME_NUM = smpl_params['poses'].shape[0]
FPS = 60


def load_human():
    smpl_params = pickle.load(open(DATA_FOLDER+'/human-params.pkl', 'rb'))  # 读取 smpl_params

    human_model = smplx.create(model_path='/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=FRAME_NUM)

    output = human_model(body_pose=smpl_params['poses'],
                         global_orient=smpl_params['orientation'],
                         transl=smpl_params['translation'])
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    # 对每一帧计算顶点法向量
    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)

    # print(f'vertices.shape: {vertices.shape}')
    # print(f'body_model.faces.shape: {human_model.faces.shape}')
    # print(f'vertex_normals.shape: {vertex_normals.shape}')

    human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

    return human_mesh


def load_object():
    object = pickle.load(open(DATA_FOLDER+'/object-params.pkl', 'rb'))  # 读取 object_params

    for key in object.keys():
        object_name = key
        decimated_mesh_path = f'../dataset/TRUMANS/Object_all/Object_mesh_decimated/{object_name}.obj'
        decimated_mesh = trimesh.load(decimated_mesh_path)
        object[key]['vertices'] = decimated_mesh.vertices
        object[key]['faces'] = decimated_mesh.faces
        object[key]['vertex_normals'] = compute_vertex_normals(object[key]['vertices'], object[key]['faces'])

    return object


def write_rerun(human: dict, object: dict):
    print('writing rerun...')
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, f'TRUMANS seg: {DATA_NAME}')
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)

    print(f'human vertices: {human["vertices"][0].shape[0]}, faces: {human["faces"].shape[0]}')
    for key in object.keys():
        print(f'{key} vertices: {object[key]["vertices"].shape[0]}, faces: {object[key]["faces"].shape[0]}')

    for i in range(FRAME_NUM):
        time = i / FPS
        rr.set_time_seconds("stable_time", time)

        rr.log(
            'human',
            rr.Mesh3D(
                vertex_positions=human['vertices'][i],
                triangle_indices=human['faces'],
                vertex_normals=human['vertex_normals'][i],
            ),
        )

        for key in object.keys():  # 一个frame中遍历所有object: 人物, 椅子, 桌子等
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

    rr.script_teardown(args)
    print('write rerun done!\n')


def main():
    human = load_human()
    object = load_object()
    write_rerun(human=human, object=object)


if __name__ == '__main__':
    main()
