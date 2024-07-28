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
from tqdm import tqdm


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


def get_sdf_grid(object_name: str, grid_dim=256, print_time=False):
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


def transform_query_inside_01cube(query: np.ndarray, sdf_info: dict):
    centroid = sdf_info['centroid']
    extents = sdf_info['extents']
    transformed_query = (query.copy() - centroid) / np.max(extents) + 0.5
    return transformed_query


def transform_query_inside_222cube(query: np.ndarray, sdf_info: dict):
    centroid = sdf_info['centroid']
    extents = sdf_info['extents']
    transformed_query = (query.copy() - centroid) / np.max(extents) * 2
    return transformed_query


def trilinear_interpolation_vectorized(data, queries):
    """
    三维双线性插值的向量化实现
    :param data: 原始数据, 形状为(a, b, c)
    :param queries: 查询点, 形状为(n, 3)，每个查询点在[0, 1]之间
    :return: 插值后的值, 形状为(n,)
    """
    a, b, c = data.shape
    n = queries.shape[0]

    # 将查询点的比例转换为实际坐标
    queries_scaled = queries * [a - 1, b - 1, c - 1]

    # 获取查询点的整数和小数部分
    queries_int = np.floor(queries_scaled).astype(int)
    queries_frac = queries_scaled - queries_int

    # 获取8个顶点的坐标
    x0, y0, z0 = queries_int[:, 0], queries_int[:, 1], queries_int[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # 边界检查，防止索引超出范围
    x0 = np.clip(x0, 0, a - 1)
    y0 = np.clip(y0, 0, b - 1)
    z0 = np.clip(z0, 0, c - 1)
    x1 = np.clip(x1, 0, a - 1)
    y1 = np.clip(y1, 0, b - 1)
    z1 = np.clip(z1, 0, c - 1)

    # 获取8个顶点的值
    f000 = data[x0, y0, z0]
    f100 = data[x1, y0, z0]
    f010 = data[x0, y1, z0]
    f110 = data[x1, y1, z0]
    f001 = data[x0, y0, z1]
    f101 = data[x1, y0, z1]
    f011 = data[x0, y1, z1]
    f111 = data[x1, y1, z1]

    # 在x方向进行线性插值
    fx00 = f000 * (1 - queries_frac[:, 0]) + f100 * queries_frac[:, 0]
    fx10 = f010 * (1 - queries_frac[:, 0]) + f110 * queries_frac[:, 0]
    fx01 = f001 * (1 - queries_frac[:, 0]) + f101 * queries_frac[:, 0]
    fx11 = f011 * (1 - queries_frac[:, 0]) + f111 * queries_frac[:, 0]

    # 在y方向进行线性插值
    fxy0 = fx00 * (1 - queries_frac[:, 1]) + fx10 * queries_frac[:, 1]
    fxy1 = fx01 * (1 - queries_frac[:, 1]) + fx11 * queries_frac[:, 1]

    # 在z方向进行线性插值
    fxyz = fxy0 * (1 - queries_frac[:, 2]) + fxy1 * queries_frac[:, 2]

    return fxyz


def query_sdf_normalized(query: np.ndarray, object_name: str):
    '''
    query 是正常 scale 的点, 坐标原点是 object translation 的中心, 不是 bounding box 的中心
    '''
    sdf = get_sdf_grid(object_name)
    sdf_grid = sdf['sdf_grid']  # (256, 256, 256)
    sdf_info = sdf['sdf_info']
    # print(f'Query point min: {np.min(query, axis=0)}, max: {np.max(query, axis=0)}')

    # # 使用 F.grid_sample 计算三线性插值
    # transformed_query = transform_query_inside_222cube(query, sdf_info)  # (k, 3), transformed_query 保证在 [-1, 1] 范围内
    # sdf_grid = torch.tensor(sdf_grid[None, None, :, :, :], dtype=torch.float32).to(config.device)
    # transformed_query = torch.tensor(transformed_query[None, :, None, None, [2, 1, 0]], dtype=torch.float32).to(config.device)
    # signed_distence = F.grid_sample(sdf_grid, transformed_query, padding_mode='border', align_corners=True)
    # signed_distence = signed_distence.cpu().numpy().reshape(-1)

    # 使用 numpy 计算三线性插值 (自己造轮子)
    transformed_query = transform_query_inside_01cube(query, sdf_info)  # (k, 3), transformed_query 保证在 [0, 1] 范围内
    signed_distence = trilinear_interpolation_vectorized(sdf_grid, transformed_query)

    return signed_distence


def visualize_sdf_grid(object_name):
    print(f'Visualizing sdf grid for {object_name}')
    rr.init(f"visualize_sdf_{object_name}", spawn=True)

    # 生成单位 [-1, 1] mesh
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

    # 生成 marching cube mesh
    sdf = get_sdf_grid(object_name)
    vertices, faces, _, _ = skimage.measure.marching_cubes(sdf['sdf_grid'], level=0)  # 返回的 mesh 大小和数组大小一样, 所以需要再次归一化
    marching_cube_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    marching_cube_mesh = get_unit_cube_mesh(marching_cube_mesh)
    marching_cube_mesh.export(f'output/marching_cube-{object_name}.obj')
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
    for _ in range(1000):
        x = random.randint(1, 256)
        y = random.randint(1, 256)
        z = random.randint(1, 256)
        coords.append((x, y, z))
    colors = np.array([[255, 0, 0] if sdf_grid[x, y, z] < 0 else [0, 255, 0] for (x, y, z) in coords])
    coords = np.array(coords)
    coords = coords/258*2-1
    colors = np.array(colors)
    rr.log('original_distence', rr.Points3D(positions=coords, colors=colors, radii=0.02))

    # 测试 query_sdf 用, 生成一些新点, 从 [-1, 1] 采样 100 个三维坐标点然后放缩到真实大小, 给出对应颜色, 大于0为绿色, 小于0为红色
    query = np.random.uniform(-1, 1, (1000, 3))
    query_original_scale = (query)*(np.max(sdf['sdf_info']['extents']))/2 + sdf['sdf_info']['centroid']
    sd = query_sdf_normalized(query_original_scale, object_name)
    colors = np.array([[255, 0, 0] if dis < 0 else [0, 255, 0] for dis in sd])
    rr.log('test_distence', rr.Points3D(positions=query, colors=colors, radii=0.02))

    return


def sample_point_cloud_from_mesh(mesh: trimesh.Trimesh, num_points: int, oversample_factor: int = 2):
    """
    从给定的3D网格中采样点云，通过过采样然后使用最远点采样进行下采样。

    参数:
    mesh (trimesh.Trimesh): 输入的3D网格对象。
    num_points (int): 需要采样的点的数量。
    oversample_factor (int): 过采样的倍数，默认为2倍。

    返回:
    np.ndarray: 采样的点云，形状为 (num_points, 3)。
    """

    # # 确保输入的网格是三角形网格
    # if not mesh.is_watertight:
    #     raise ValueError("输入的网格必须是闭合且 watertight 的三角形网格。")

    # 过采样点数
    oversample_points = num_points * oversample_factor

    # 从网格中采样点
    points, _ = trimesh.sample.sample_surface_even(mesh, oversample_points)

    # 使用最远点采样进行下采样
    selected_points = farthest_point_sampling(points, num_points)

    return selected_points


def farthest_point_sampling(points, num_samples):
    """
    使用最远点采样从点云中选择指定数量的点。

    参数:
    points (np.ndarray): 输入的点云，形状为 (B, N, 3) 或者 (N, 3)
    num_samples (int): 需要采样的点的数量。

    返回:
    np.ndarray: 采样的点云，形状为 (B, num_samples, 3) 或者 (num_samples, 3)
    """
    if len(points.shape) == 2:
        points = np.expand_dims(points, axis=0)
    B, N, D = points.shape

    # 初始化返回的采样点索引
    centroids = np.zeros((B, num_samples), dtype=np.int32)
    # 初始化最大距离数组
    distances = np.ones((B, N)) * 1e10
    # 随机选择初始点
    farthest = np.random.randint(0, N, B)
    batch_indices = np.arange(B)

    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest].reshape(B, 1, D)
        dist = np.sum((points - centroid) ** 2, -1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances, axis=1)

    sampled_points = points[batch_indices[:, None], centroids]
    if B == 1:
        sampled_points = sampled_points[0]
    return sampled_points


def main():
    # 测试 sdf_grid 是否正确生成
    object_list = os.listdir(config.OBJECT_ORIGINAL_PATH)
    for object_name in object_list:
        # visualize_sdf_grid(object_name.split('.')[0])
        # exit(0)
        get_sdf_grid(object_name.split('.')[0])
    return


if __name__ == '__main__':
    main()
