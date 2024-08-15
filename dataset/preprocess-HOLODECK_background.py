'''
Add motion to HOLODECK scene
Input: HOLODECK_NAME
Output: HOLODECK/{HOLODECK_NAME}/motion.pkl & *.rrd
同时可视化物体mesh点云、房间和人
'''
import pickle
import gzip
import rerun as rr
import numpy as np
from collections import Counter
import os
import config
import json
from utils.mesh_utils import compute_vertex_normals
from scipy.spatial.transform import Rotation as R
import smplx
import torch
import trimesh

HOLODECK_NAME = 'a_DiningRoom_with_round_table_-2024-08-07-14-52-49-547177'
# HOLODECK_NAME = 'a_DiningRoom-2024-08-07-14-59-21-965489'
# HOLODECK_NAME = 'a_living_room-2024-08-07-14-16-58-057245'
json_name = HOLODECK_NAME.split('-')[0]
save_rrd = False


def set_up_rerun():
    # start rerun script
    rr.init(f'Visualization: {HOLODECK_NAME}', spawn=not save_rrd)
    if save_rrd:
        rr.save(os.path.join('rerun', f'{json_name}.rrd'))
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)


def write_objects():
    '''
    write objects to rerun
    Output: bounding box of this scene
    '''
    # write objects
    holodeck_path = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, f'{json_name}.json')
    with open(holodeck_path, 'r') as f:
        holodeck_info = json.load(f)

    bb_min = np.array([np.inf, np.inf, np.inf])
    bb_max = np.array([-np.inf, -np.inf, -np.inf])
    for obj_info in holodeck_info['objects']:
        obj_name = obj_info['assetId']  # 一串乱码
        obj_id = obj_info['id']  # 一个英文名, 有含义
        if '|' in obj_id:
            continue
        obj_path = os.path.join(config.OBJATHOR_BASE, obj_name, f'{obj_name}.pkl.gz')
        if not os.path.exists(obj_path):  # skip non-exist obj
            continue
        obj_translation = np.array([obj_info['position']['x'], obj_info['position']['y'], obj_info['position']['z']])
        obj_orientation = np.array([obj_info['rotation']['x'], obj_info['rotation']['y'], obj_info['rotation']['z']])
        R_obj_rot = R.from_euler('XYZ', obj_orientation, degrees=True)

        with gzip.open(obj_path, 'rb') as f:
            obj = pickle.load(f)
        obj_y_rot_offset = obj['yRotOffset']
        R_obj_y_rot_offset = R.from_euler('Y', obj_y_rot_offset, degrees=True)
        # print(f'id: {obj_id}, y_rot_offset: {obj_y_rot_offset}')

        vertices = np.array([[v['x'], v['y'], v['z']] for v in obj['vertices']])
        vertices = (R_obj_rot*R_obj_y_rot_offset).apply(vertices) + obj_translation
        triangles = np.array([[obj['triangles'][i], obj['triangles'][i+1], obj['triangles'][i+2]] for i in range(0, len(obj['triangles']), 3)])

        # 测试用
        if 'chair' in obj_id:
            chair_path = os.path.join(config.OBJECT_ORIGINAL_PATH, 'static_chair_03.obj')
            mesh = trimesh.load_mesh(chair_path)
            vertices = mesh.vertices
            R_y90 = R.from_euler('Y', -90, degrees=True)
            vertices = (R_y90*R_obj_rot).apply(vertices) + obj_translation + np.array([0, 0.45, 0])
            triangles = mesh.faces
        # 测试用

        normals = compute_vertex_normals(vertices, triangles)

        rr.log(
            f'object/{obj_id.replace(" ", "-")}',
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                vertex_normals=normals,
            ),
        )

        bb_min = np.minimum(bb_min, np.min(vertices, axis=0))
        bb_max = np.maximum(bb_max, np.max(vertices, axis=0))

    print(f'bb_min: {bb_min}, bb_max: {bb_max}')

    rr.log('bbox',
           rr.Boxes3D(
               centers=(bb_min + bb_max) / 2,
               half_sizes=(bb_max - bb_min) / 2,
               radii=0.01,
               colors=(0, 0, 255),
           ))

    rr.log('choose-bbox',
           rr.Boxes3D(
               centers=[3, 1, 4]+bb_min,
               half_sizes=[3, 1, 4],
               radii=0.01,
               colors=(255, 0, 0),
           ))

    return bb_min


def write_human(bb_min):
    # write human
    motion_path = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, 'motion.pkl')
    if not os.path.exists(motion_path):  # skip non-exist motion
        print(f'motion file not exist: {motion_path}')
        return
    with open(motion_path, 'rb') as f:
        motion = pickle.load(f)

    frame_interval = 5
    fps = 30 / frame_interval
    motion_translation = motion['translation'][::frame_interval]+bb_min
    motion_orientation = motion['orientation'][::frame_interval]
    motion_poses = motion['poses'][::frame_interval]
    max_frame_num = motion_poses.shape[0]

    human_model = smplx.create(model_path=config.SMPL_MODEL_PATH,
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=max_frame_num)

    # print(f'body pose: {motion_pose[:,1:].shape}')
    # print(f'global orient: {motion_pose[:,0].shape}')
    # print(f'translation: {motion_translation.shape}')
    output = human_model(body_pose=torch.tensor(motion_poses, dtype=torch.float32),
                         global_orient=torch.tensor(motion_orientation, dtype=torch.float32),
                         transl=torch.tensor(motion_translation, dtype=torch.float32))
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    for i in range(max_frame_num):
        time = i / fps
        rr.set_time_seconds("stable_time", time)

        rr.log(
            'human',
            rr.Mesh3D(
                vertex_positions=vertices[i],
                triangle_indices=faces,
                vertex_normals=compute_vertex_normals(vertices[i], faces),
            ),
        )


def get_object_in_background(bb_min):
    '''
    输出一个 (300, 100, 400) 的背景数组，其中包含了所有的物体的 True 区域
    只要和地板和墙壁取并集即可得到最终的背景数组
    '''
    # get obj info
    holodeck_path = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, f'{json_name}.json')
    with open(holodeck_path, 'r') as f:
        holodeck_info = json.load(f)

    background = np.zeros((300, 100, 400), dtype=bool)
    for obj_info in holodeck_info['objects']:
        obj_name = obj_info['assetId']
        obj_id = obj_info['id']
        if '|' in obj_id:
            continue
        if 'chair' not in obj_id:  # skip non-chair, test for TRUMANS
            print(f'chair: {obj_id}, assetId: {obj_name}')
            continue

        cache_path = os.path.join(config.DATA_OBJATHOR_CACHE_PATH, f'{obj_name}.npy')
        if os.path.exists(cache_path):  # load cache
            with open(cache_path, 'rb') as f:
                inside_points = np.load(f)
            print(f'obj_name: {obj_id}, inside_points num: {inside_points.shape}')
        else:  # compute inside points, not exist in cache
            obj_path = os.path.join(config.OBJATHOR_BASE, obj_name, f'{obj_name}.pkl.gz')

            if not os.path.exists(obj_path):  # skip non-exist obj
                continue
            with gzip.open(obj_path, 'rb') as f:  # load obj mesh
                obj = pickle.load(f)

            obj_y_rot_offset = obj['yRotOffset']
            R_obj_y_rot_offset = R.from_euler('Y', obj_y_rot_offset, degrees=True)

            vertices = np.array([[v['x'], v['y'], v['z']] for v in obj['vertices']])
            vertices = (R_obj_y_rot_offset).apply(vertices)
            triangles = np.array([[obj['triangles'][i], obj['triangles'][i+1], obj['triangles'][i+2]] for i in range(0, len(obj['triangles']), 3)])
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)  # create mesh

            # 测试用, 用新的直接替代之前的椅子, 之后的都一样
            if 'chair' in obj_id:
                print(f'using new chair: {obj_id}, assetId: {obj_name}')
                chair_path = os.path.join(config.OBJECT_ORIGINAL_PATH, 'static_chair_03.obj')
                mesh = trimesh.load_mesh(chair_path)
                vertices = mesh.vertices
                R_y90 = R.from_euler('Y', -90, degrees=True)
                vertices = (R_y90).apply(vertices) + np.array([0, 0.45, 0])
                triangles = mesh.faces
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            # 测试用

            print(f'name: {obj_id}, mesh bounding box: {mesh.bounds}, points: ~16000, obj_key: {obj.keys()}')

            bbox = mesh.bounds  # get mesh bounding box
            x_coords = np.arange(bbox[0, 0], bbox[1, 0], 0.02, dtype=float)
            y_coords = np.arange(bbox[0, 1], bbox[1, 1], 0.02, dtype=float)
            z_coords = np.arange(bbox[0, 2], bbox[1, 2], 0.02, dtype=float)
            x, y, z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)  # create points, a 3D grid, 0.02m resolution
            print(f'points num: {points.shape}')

            ray_origins = points  # ray origins
            ray_directions = np.array([[0, 1, 0]]*len(ray_origins))  # ray directions

            mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)  # build ray-mesh intersector
            # locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)  # get intersection points
            locations, index_ray, index_tri = mesh_intersector.intersects_location(ray_origins, ray_directions)

            # 使用Counter统计每个元素的出现次数
            count = Counter(index_ray)

            # 将Counter结果转化为两个numpy数组: 元素数组和对应的出现次数数组
            elements = np.array(list(count.keys()))
            frequencies = np.array(list(count.values()))

            # 筛选出出现奇数次的元素, 即在物体内部的点的索引
            odd_occurrence_mask = (frequencies % 2 != 0)
            odd_occurrence_elements = elements[odd_occurrence_mask]
            inside_points = points[odd_occurrence_elements]
            print(f'obj_name: {obj_id}, inside_points num: {inside_points.shape}')
            np.save(cache_path, inside_points)

        obj_translation = np.array([obj_info['position']['x'], obj_info['position']['y'], obj_info['position']['z']])
        obj_orientation = np.array([obj_info['rotation']['x'], obj_info['rotation']['y'], obj_info['rotation']['z']])
        R_obj_rot = R.from_euler('XYZ', obj_orientation, degrees=True)

        inside_points = (R_obj_rot).apply(inside_points) + obj_translation  # rotate and translate points, based on HOLODECK output
        inside_points -= bb_min  # shift to the (0, 0, 0) based room
        x_idx = np.clip((inside_points[:, 0] * 50).astype(int), 0, 299)  # clip to the room size
        y_idx = np.clip((inside_points[:, 1] * 50).astype(int), 0, 99)
        z_idx = np.clip((inside_points[:, 2] * 50).astype(int), 0, 399)
        background[x_idx, y_idx, z_idx] = True

        rr.log(f'obj_point/{obj_id.replace(" ", "-")}', rr.Points3D(positions=inside_points + bb_min))
        print(f'middle point: {(np.max(inside_points, axis=0) + np.min(inside_points, axis=0))/2}')

    return background


def get_background(bb_min):
    '''
    计算 (300, 100, 400) 的背景数组, 然后保存到背景文件中
    所有物体有可能和 (0, 0, 0) 的基准房间有偏移, 所以需要把物体的坐标减去整个房间的 bounding box 的最小值
    '''
    background = np.zeros((300, 100, 400), dtype=bool)

    # 加入地板和墙壁
    background[:, :5, :] = True
    background[:5, :, :] = True
    background[-5:, :, :] = True
    background[:, :, :5] = True
    background[:, :, -5:] = True

    # 加入物体
    object_background = get_object_in_background(bb_min)
    background = np.logical_or(background, object_background)

    background_path = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, 'background.npy')
    np.save(background_path, background)

    return


def main():
    set_up_rerun()
    bb_min = write_objects()
    write_human(bb_min)
    get_background(bb_min)


if __name__ == '__main__':
    main()
