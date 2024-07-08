import json
import time
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface


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


def generate_sdf(mesh, dest_json_path, dest_sdf_path, dest_voxel_mesh_path="", grid_dim=256, print_time=True):
    """
    生成一个Signed Distance Field (SDF)从一个给定的三角网格对象，并保存相关数据。

    参数:
        mesh:                   输入的trimesh对象
        dest_json_path:         输出JSON文件路径
        dest_sdf_path:          输出SDF文件路径
        dest_voxel_mesh_path:   输出体素网格（obj文件）路径；默认为空字符串，表示不会保存体素网格
        grid_dim:               SDF的网格维度（N）；默认值为256
        print_time:             如果为True，打印生成SDF所花费的时间；默认值为True

    返回:
        centroid:               SDF的质心
        extents:                SDF的范围
        sdf:                    包含SDF值的N x N x N的numpy数组
    """

    # 计算并保存质心和范围数据，用于将顶点转换为[-1,1]范围内
    centroid = mesh.bounding_box.centroid
    extents = mesh.bounding_box.extents

    # 将质心和范围数据保存为JSON格式
    json_dict = {
        'centroid': centroid.tolist(),
        'extents': extents.tolist(),
        'grid_dim': grid_dim
    }
    with open(dest_json_path, 'w') as json_file:
        json.dump(json_dict, json_file)

    if print_time:
        start_time = time.time()

    # 使用mesh_to_voxels函数生成SDF
    sdf = mesh_to_voxels(mesh, voxel_resolution=grid_dim)

    # 如果需要保存体素网格
    if dest_voxel_mesh_path:
        import skimage.measure
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
        voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        voxel_mesh.export(dest_voxel_mesh_path, file_type='obj')

    if print_time:
        print(f"Generating SDF took {time.time() - start_time} seconds")

    # 保存SDF数据为.npy文件
    np.save(dest_sdf_path, sdf)

    return centroid, extents, sdf

# 使用示例
# mesh = trimesh.load('path_to_mesh.obj')
# centroid, extents, sdf = generate_sdf(mesh, 'sdf_data.json', 'sdf_values.npy')
