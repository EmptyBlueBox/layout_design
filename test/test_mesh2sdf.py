import numpy as np
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf
from mesh_to_sdf import mesh_to_voxels  # The mesh_to_voxels function creates an N ✕ N ✕ N array of SDF values.
import pyrender

import trimesh
import skimage

mesh = trimesh.load('chair.obj')
mesh = trimesh.load('/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS/Object_all/Object_mesh/drawer_base_01.obj')


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


# 1
voxels = mesh_to_voxels(mesh, 256, pad=True, sign_method="depth")
print(f'voxels.shape: {voxels.shape}')

vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
mesh.show()


# # 2
# points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)

# print(f'points.shape: {points.shape}, sdf.shape: {sdf.shape}')
# print(f'points: {points[:5]}, real sdf: {sdf[:5]}')

# query_points = points[:5]
# sdf = mesh_to_sdf(get_unit_sphere_mesh(mesh), query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None,
#                   scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
# print(f'test sdf: {sdf}')

# colors = np.zeros(points.shape)
# colors[sdf < 0, 2] = 1
# colors[sdf > 0, 0] = 1
# cloud = pyrender.Mesh.from_points(points, colors=colors)
# scene = pyrender.Scene()
# scene.add(cloud)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
