import smplx
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from rerun.datatypes import Quaternion, Angle, RotationAxisAngle
import rerun as rr
import trimesh
import pickle
import torch
from utils.mesh_utils import compute_vertex_normals
from utils.pytorch3d_utils import quaternion_multiply, axis_angle_to_quaternion, quaternion_to_axis_angle, quaternion_apply
import config
import os


def write_object_human(object_params,
                       human_params,
                       delta_x,
                       delta_theta,
                       object_mesh_path=config.OBJECT_DECIMATED_PATH,
                       max_frame_num=500,
                       fps=30):
    R_delta = R.from_rotvec(np.array([0, delta_theta, 0]))
    frame_num = human_params['poses'].shape[0]
    # 计算基础物体旋转之后的参数和顶点, 自己旋转即可
    object_base = next(iter(object_params.keys()))
    object_mesh = {}
    for key in object_params.keys():
        object_orientation = R_delta*R.from_euler('xyz', object_params[key]['rotation'])
        if key == object_base:
            object_translation = object_params[key]['location'] + delta_x
        else:
            object_translation = R_delta.apply(object_params[key]['location']-object_params[object_base]
                                               ['location'])+object_params[object_base]['location']+delta_x
        decimated_mesh_path = os.path.join(object_mesh_path, key + '.obj')
        decimated_mesh = trimesh.load(decimated_mesh_path)
        object_mesh[key] = {}  # 初始化一个子物体的字典
        object_mesh[key]['vertices'] = np.zeros((max_frame_num, decimated_mesh.vertices.shape[0], 3))
        for t in range(frame_num):
            object_mesh[key]['vertices'][t] = object_orientation[t].apply(decimated_mesh.vertices) + object_translation[t]
        for t in range(frame_num, max_frame_num):
            object_mesh[key]['vertices'][t] = object_mesh[key]['vertices'][frame_num-1]
        object_mesh[key]['faces'] = decimated_mesh.faces
        object_mesh[key]['vertex_normals'] = np.zeros_like(object_mesh[key]['vertices'])
        for t in range(max_frame_num):
            object_mesh[key]['vertex_normals'][t] = compute_vertex_normals(object_mesh[key]['vertices'][t], object_mesh[key]['faces'])

    object_base = next(iter(object_params.keys()))
    # 计算人体旋转之后的参数
    human_orientation = R_delta*R.from_rotvec(human_params['orientation'])
    human_params['orientation'] = human_orientation.as_rotvec()
    human_params['translation'] = R_delta.apply(
        human_params['translation']-object_params[object_base]['location'])+object_params[object_base]['location']+delta_x
    # 计算人体的顶点
    human_model = smplx.create(model_path=config.SMPL_MODEL_PATH,
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=frame_num)
    output = human_model(body_pose=torch.tensor(human_params['poses'], dtype=torch.float32),
                         global_orient=torch.tensor(human_params['orientation'], dtype=torch.float32),
                         transl=torch.tensor(human_params['translation'], dtype=torch.float32))
    vertices_0 = output.vertices.detach().cpu().numpy()
    vertices = np.zeros((max_frame_num, vertices_0.shape[1], 3))
    vertices[:frame_num] = vertices_0
    vertices[frame_num:] = vertices[frame_num-1]
    faces = human_model.faces
    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)
    human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

    for i in range(max_frame_num):
        time = i / fps
        rr.set_time_seconds("stable_time", time)

        # 画人
        rr.log(
            f'{object_base}/human',
            rr.Mesh3D(
                vertex_positions=human_mesh['vertices'][i],
                triangle_indices=human_mesh['faces'],
                vertex_normals=human_mesh['vertex_normals'][i],
            ),
        )

        # 画物体
        for key in object_mesh.keys():  # 一个frame中遍历所有object: 人物, 椅子, 桌子等
            rr.log(f'{object_base}/object/{key}',
                   rr.Mesh3D(
                       vertex_positions=object_mesh[key]['vertices'][i],
                       triangle_indices=object_mesh[key]['faces'],
                       vertex_normals=object_mesh[key]['vertex_normals'][i],
                   ),
                   )
