'''
Input: motion and wall parameter
Output: chair rotation
Optimize parameter: chair rotation
Optimization method: grid search
'''

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


DATA_NAME = 'seat_5-frame_num_150'
EPOCH = 200

DATA_FOLDER = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/{DATA_NAME}'
smpl_params = pickle.load(open(DATA_FOLDER+'/human-params.pkl', 'rb'))  # 读取 smpl_params
FRAME_NUM = smpl_params['poses'].shape[0]
FPS = 60
PARAMETER_WALL = {
    'centers': [0, .9, 1.9],
    'half_sizes': [2., 1., 1.5],
    'axis': [0, 1, 0],
    'angle': -15,
}
PARAMETER_WALL = {
    'centers': [-0.3, .9, 1.9],
    'half_sizes': [.3, 1., 2.],
    'axis': [0, 1, 0],
    'angle': 0,
}


def load_human():
    human_params = pickle.load(open(DATA_FOLDER+'/human-params.pkl', 'rb'))  # 读取 smpl_params

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


def write_rerun(human: dict, object: dict, seg_name: str):
    print('writing rerun...')
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, f'gradient descent vanilla: {DATA_NAME}')
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)

    print(f'human vertices: {human["vertices"][0].shape[0]}, faces: {human["faces"].shape[0]}')
    for key in object.keys():
        print(f'{key} vertices: {object[key]["vertices"].shape[0]}, faces: {object[key]["faces"].shape[0]}')

    # 画墙
    rr_transform = rr.Transform3D(
        rotation=RotationAxisAngle(axis=PARAMETER_WALL['axis'], angle=Angle(deg=PARAMETER_WALL['angle'])),
    )
    rr.log(f"{seg_name}/wall", rr_transform)
    rr.log(
        f"{seg_name}/wall",
        rr.Boxes3D(
            centers=PARAMETER_WALL['centers'],
            half_sizes=PARAMETER_WALL['half_sizes'],
            radii=0.01,
            colors=(0, 0, 255),
            labels="blue",
        ),
    )

    for i in range(FRAME_NUM):
        time = i / FPS
        rr.set_time_seconds("stable_time", time)

        # 画人
        rr.log(
            f'{seg_name}/human',
            rr.Mesh3D(
                vertex_positions=human['vertices'][i],
                triangle_indices=human['faces'],
                vertex_normals=human['vertex_normals'][i],
            ),
        )

        # 画物体
        for key in object.keys():  # 一个frame中遍历所有object: 人物, 椅子, 桌子等
            rr_transform = rr.Transform3D(
                rotation=Quaternion(xyzw=R.from_euler('xyz', object[key]['rotation'][i]).as_quat()),
                translation=object[key]['location'][i],
            )

            rr.log(f'{seg_name}/object/{key}', rr_transform)
            rr.log(f'{seg_name}/object/{key}',
                   rr.Mesh3D(
                       vertex_positions=object[key]['vertices'],
                       triangle_indices=object[key]['faces'],
                       vertex_normals=object[key]['vertex_normals'],
                   ),
                   )

    rr.script_teardown(args)
    print('write rerun done!\n')


def object_function(vertices: torch.Tensor):
    # 计算平面方程
    normal_vector = np.array([0, 0, PARAMETER_WALL['half_sizes'][2]])  # 墙是竖直的, 法向量是(0, 0, 1.5)
    wall_center_point = np.array(PARAMETER_WALL['centers'])  # 墙的中心点
    normal_vector = R.from_rotvec(PARAMETER_WALL['angle'] * np.pi / 180 * np.array([0, 1, 0])).apply(normal_vector)  # 法向量旋转
    wall_center_point -= normal_vector  # 墙的中心点也要旋转

    # print(f'normal_vector: {normal_vector}')
    # print(f'wall_center_point: {wall_center_point}')

    # 将numpy数组转换为torch张量以便与输入的vertices进行运算
    normal_vector = torch.tensor(normal_vector, dtype=torch.float32)
    wall_center_point = torch.tensor(wall_center_point, dtype=torch.float32)

    # 计算每个顶点到墙平面的距离
    # 点到平面距离公式：distance = (point - wall_center_point) . normal_vector / ||normal_vector||
    distances = torch.matmul(vertices - wall_center_point, normal_vector) / torch.norm(normal_vector)

    # 只考虑正数的距离
    positive_distances = distances[distances > 0]

    print(f'penetrate points number: {(distances > 0).sum()}')

    # 计算距离的平方和
    loss = torch.sum(positive_distances ** 2)

    return loss


def optimize():
    human_params = pickle.load(open(DATA_FOLDER+'/human-params.pkl', 'rb'))  # 读取 smpl_params
    human_model = smplx.create(model_path='/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=1)

    object = pickle.load(open(DATA_FOLDER+'/object-params.pkl', 'rb'))  # 读取 object_params

    vector_chair_to_human = torch.tensor(human_params['translation'][-1] - object['movable_chair_seat_01']['location'][-1], dtype=torch.float32)

    PARAMETER_OPTIMIZE = {
        'orientation': torch.zeros(1, dtype=torch.float32, requires_grad=True),
    }

    # 优化
    optimizer = torch.optim.Adam([PARAMETER_OPTIMIZE['orientation']], lr=0.1)
    for i in range(EPOCH):
        delta_quaternion = axis_angle_to_quaternion(torch.tensor([0, 1, 0])*PARAMETER_OPTIMIZE['orientation'])
        new_global_orient = quaternion_to_axis_angle(quaternion_multiply(
            delta_quaternion, axis_angle_to_quaternion(torch.tensor(human_params['orientation'][-1], dtype=torch.float32))))
        new_transl = torch.tensor(object['movable_chair_seat_01']['location'][-1]) + \
            quaternion_apply(delta_quaternion, vector_chair_to_human)
        output = human_model(body_pose=torch.tensor(human_params['poses'][-1, None, :], dtype=torch.float32),
                             global_orient=new_global_orient[None, :],
                             transl=new_transl[None, :])
        vertices = output.vertices
        # 计算 loss
        loss = object_function(vertices)
        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch: {i}, loss: {loss}\n')

    # 优化完成, 可视化
    # 可视化人
    delta_quaternion = axis_angle_to_quaternion(torch.tensor([0, 1, 0])*PARAMETER_OPTIMIZE['orientation'])

    human_model = smplx.create(model_path='/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=FRAME_NUM)

    human_params['orientation'] = quaternion_to_axis_angle(quaternion_multiply(
        delta_quaternion, axis_angle_to_quaternion(torch.tensor(human_params['orientation'], dtype=torch.float32))))  # 更新人的朝向
    human_params['translation'] = torch.tensor(object['movable_chair_seat_01']['location']) + \
        quaternion_apply(delta_quaternion, torch.tensor(human_params['translation'] - object['movable_chair_seat_01']['location']))  # 更新人的位置
    output = human_model(body_pose=torch.tensor(human_params['poses'], dtype=torch.float32),
                         global_orient=human_params['orientation'],
                         transl=human_params['translation'])
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)

    human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

    # 可视化物体
    delta = R.from_rotvec(np.array([0, 1, 0])*PARAMETER_OPTIMIZE['orientation'].detach().numpy())
    for key in object.keys():
        object_name = key
        for i in range(FRAME_NUM):
            object[key]['rotation'][i] = (delta*R.from_euler('xyz', object[key]['rotation'][i])).as_euler('xyz')  # 更新物体的朝向
        decimated_mesh_path = f'../dataset/TRUMANS/Object_all/Object_mesh_decimated/{object_name}.obj'
        decimated_mesh = trimesh.load(decimated_mesh_path)
        object[key]['vertices'] = decimated_mesh.vertices
        object[key]['faces'] = decimated_mesh.faces
        object[key]['vertex_normals'] = compute_vertex_normals(object[key]['vertices'], object[key]['faces'])

    return human_mesh, object


def main():
    human = load_human()
    object = load_object()
    write_rerun(human=human, object=object, seg_name='before')

    human, object = optimize()
    write_rerun(human=human, object=object, seg_name='after')


if __name__ == '__main__':
    main()
