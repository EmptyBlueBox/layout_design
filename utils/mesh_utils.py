from asyncio import events
import json
import os
import time
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
import config
import torch
import torch.nn.functional as F
import rerun as rr
import skimage
import random


def compute_vertex_normals(vertices, faces):
    """
    使用向量化操作计算顶点法向量。

    参数:
    vertices (np.ndarray): 顶点坐标数组，形状为 (N, 3)。
    faces (np.ndarray): 面的顶点索引数组，形状为 (M, 3)。

    返回:
    np.ndarray: 归一化后的顶点法向量数组，形状为 (N, 3)。
    """
    # 获取三角形的顶点
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 计算每个面的法向量
    normals = np.cross(v1 - v0, v2 - v0)

    # 计算法向量的长度
    norm_lengths = np.linalg.norm(normals, axis=1)

    # 避免除以零，将长度为零的法向量设为一个微小值
    norm_lengths[norm_lengths == 0] = 1e-10

    # 归一化法向量
    normals /= norm_lengths[:, np.newaxis]

    # 将法向量累加到顶点上
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], normals)

    # 计算顶点法向量的长度
    vertex_norm_lengths = np.linalg.norm(vertex_normals, axis=1)

    # 避免除以零，将长度为零的顶点法向量设为一个微小值
    vertex_norm_lengths[vertex_norm_lengths == 0] = 1e-10

    # 归一化顶点法向量
    vertex_normals = (vertex_normals.T / vertex_norm_lengths).T
    return vertex_normals


def get_unit_sphere_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def get_unit_cube_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def get_sdf_grid(object_name: str, grid_dim=256, print_time=True):
    if print_time:
        start_time = time.time()

    # get sdf grid
    if not os.path.exists(config.OBJECT_SDF_GRID_PATH):
        os.makedirs(config.OBJECT_SDF_GRID_PATH)

    sdf_grid_list = os.listdir(config.OBJECT_SDF_GRID_PATH)
    sdf_grid_path = os.path.join(config.OBJECT_SDF_GRID_PATH, f'{object_name}.npy')
    obj_path = os.path.join(config.OBJECT_ORIGINAL_PATH, f'{object_name}.obj')

    if f'{object_name}.npy' in sdf_grid_list:
        sdf_grid = np.load(sdf_grid_path)
    else:
        mesh = trimesh.load(obj_path)
        sdf_grid = mesh_to_voxels(mesh, voxel_resolution=grid_dim, pad=True, sign_method="depth")
        np.save(sdf_grid_path, sdf_grid)

    # get sdf info
    if not os.path.exists(config.OBJECT_SDF_INFO_PATH):
        os.makedirs(config.OBJECT_SDF_INFO_PATH)

    sdf_info_list = os.listdir(config.OBJECT_SDF_INFO_PATH)
    sdf_info_path = os.path.join(config.OBJECT_SDF_INFO_PATH, f'{object_name}.json')

    if f'{object_name}.json' in sdf_info_list:
        with open(sdf_info_path, 'r') as json_file:
            sdf_info = json.load(json_file)
    else:
        mesh = trimesh.load(obj_path)
        # 计算并保存质心和范围数据，用于将顶点转换为[-1,1]范围内
        centroid = mesh.bounding_box.centroid
        extents = mesh.bounding_box.extents

        # 将质心和范围数据保存为JSON格式
        sdf_info = {
            'grid_dim': grid_dim,  # 256
            'centroid': centroid.tolist(),  # [x, y, z]
            'extents': extents.tolist(),  # [x, y, z]
        }
        with open(sdf_info_path, 'w') as json_file:
            json.dump(sdf_info, json_file)

    if print_time:
        print(f"Generating SDF took {time.time() - start_time} seconds")

    output = {
        'sdf_grid': sdf_grid,  # 256x256x256 array
        'sdf_info': sdf_info,  # dict from JSON file
    }
    return output


def transform_query_with_unit_cube(query: np.ndarray, sdf_info: dict):
    centroid = sdf_info['centroid']
    extents = sdf_info['extents']
    transformed_query = (query.copy() - centroid) / (np.max(extents) * 0.5)
    return transformed_query


def query_sdf_normalized(query: np.ndarray, object_name: str):
    sdf = get_sdf_grid(object_name)
    sdf_grid = sdf['sdf_grid']  # (256, 256, 256)
    sdf_info = sdf['sdf_info']
    transformed_query = transform_query_with_unit_cube(query, sdf_info)  # (k, 3), query 保证在 [-1, 1] 范围内

    sdf_grid = torch.tensor(sdf_grid[None, None, :, :, :], dtype=torch.float32).to(config.device)
    transformed_query = torch.tensor(transformed_query[None, :, None, None, :], dtype=torch.float32).to(config.device)
    sd = F.grid_sample(sdf_grid, transformed_query, mode='bilinear', padding_mode='border', align_corners=True)
    sd = sd.cpu().numpy().reshape(-1)

    return sd


def visualize_sdf_grid(object_name):
    rr.init("visualize_sdf", spawn=True)

    obj_path = os.path.join(config.OBJECT_ORIGINAL_PATH, f'{object_name}.obj')
    mesh = trimesh.load(obj_path)
    unit_mesh = get_unit_cube_mesh(mesh)  # 可视化的都是单位立方体内的 mesh
    rr.log(
        "obj_mesh",
        rr.Mesh3D(
            vertex_positions=unit_mesh.vertices,
            triangle_indices=unit_mesh.faces,
            vertex_normals=compute_vertex_normals(unit_mesh.vertices, unit_mesh.faces),
        ),
    )
    print(f'original bounding_box scale: {np.max(unit_mesh.vertices, axis=0)-np.min(unit_mesh.vertices, axis=0)}')

    sdf = get_sdf_grid(object_name)
    vertices, faces, _, _ = skimage.measure.marching_cubes(sdf['sdf_grid'], level=0)  # 返回的 mesh 大小和数组大小一样, 所以需要再次归一化
    marching_cube_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    marching_cube_mesh = get_unit_cube_mesh(marching_cube_mesh)
    rr.log("marching_cube_sdf_grid", rr.Transform3D(translation=unit_mesh.bounding_box.extents))
    rr.log(
        "marching_cube_sdf_grid",
        rr.Mesh3D(
            vertex_positions=marching_cube_mesh.vertices,
            triangle_indices=marching_cube_mesh.faces,
            vertex_normals=compute_vertex_normals(marching_cube_mesh.vertices, marching_cube_mesh.faces),
        ),
    )
    print(f'marching cube bounding_box scale: {np.max(marching_cube_mesh.vertices, axis=0)-np.min(marching_cube_mesh.vertices, axis=0)}')

    # 测试 get_sdf_grid 用, 生成sdf使用的点
    print(f"inside number: {np.sum(sdf['sdf_grid'] < 0)}, outside number: {np.sum(sdf['sdf_grid'] > 0)}")
    sdf_grid = sdf['sdf_grid']  # (256, 256, 256)
    coords = []  # 从 sdf_grid 随机采样 100 个三维坐标点, 给出对应颜色, 大于0为绿色, 小于0为红色
    for _ in range(100):
        x = random.randint(0, 255)
        y = random.randint(0, 255)
        z = random.randint(0, 255)
        coords.append((x, y, z))
    colors = []
    for (x, y, z) in coords:
        if sdf_grid[x, y, z] > 0:
            colors.append([0, 255, 0])
        else:
            colors.append([255, 0, 0])
    coords = np.array(coords)
    coords = coords/256*2-1
    colors = np.array(colors)
    rr.log('original_distence', rr.Points3D(positions=coords, colors=colors, radii=0.02))

    # 测试 query_sdf 用, 生成一些新点
    # 从 [-1, 1] 采样 100 个三维坐标点, 给出对应颜色, 大于0为绿色, 小于0为红色
    query = np.random.uniform(-1, 1, (100, 3))
    query_original_scale = query*(np.max(sdf['sdf_info']['extents']) * 0.5) + sdf['sdf_info']['centroid']
    sd = query_sdf_normalized(query_original_scale, object_name)
    print(f'Query point: {query[-5:]}\nSigned distence: {sd[-5:]}')
    colors = np.array([[255, 0, 0] if dis < 0 else [0, 255, 0] for dis in sd])
    rr.log('test_distence', rr.Points3D(positions=query, colors=colors, radii=0.02))

    return


def main():
    object_name = 'drawer_base_01'
    visualize_sdf_grid(object_name)
    return


if __name__ == '__main__':
    main()
