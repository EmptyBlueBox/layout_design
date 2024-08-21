'''
建构一个 Kitchen , 包含架子、冰箱、微波炉和工作台, 架子上面有箱子
使用 Rerun 可视化
'''

import rerun as rr
import os
import numpy as np
import trimesh
import config
import pickle
import torch
import smplx
from utils.mesh_utils import compute_vertex_normals
from scipy.spatial.transform import Rotation as R

room_config_vanilla = {
    'room': {
        'x_min': 0,
        'x_max': 6,
        'y_min': 0,
        'y_max': 3,
        'z_min': 0,
        'z_max': 6,
    },
    'objects': {
        'shelf-0': {
            'translation': [5, 0, 1],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'shelf-1': {
            'translation': [5, 0, 3],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'shelf-2': {
            'translation': [5, 0, 5],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'wall_mounted_shelf-0': {
            'translation': [-2, 1.5, 1],
            'scale': [1, 1, 1],
            'orientation': -90,
        },
        'wall_mounted_shelf-1': {
            'translation': [-2, 1.5, 3],
            'scale': [1, 1, 1],
            'orientation': -90,
        },
        'wall_mounted_shelf-2': {
            'translation': [-2, 1.5, 5],
            'scale': [1, 1, 1],
            'orientation': -90,
        },
        'fridge_base': {
            'translation': [3, 0, 8],
            'scale': [1, 1, 1],
            'orientation': 90,
        },
        'table': {
            'translation': [2.5, 0, 3],
            'scale': [1, 1, 1],
            'orientation': 90,
        },
        'food-0': {
            'translation': [5, 0.39, 1],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-1': {
            'translation': [5, 0.74, 3],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-2': {
            'translation': [5, 1.09, 5],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-3': {
            'translation': [0.2, 1.27, 1],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-4': {
            'translation': [0.2, 1.83, 3],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-5': {
            'translation': [0.2, 2.15, 5],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-6': {
            'translation': [3, 0.6, 5.6],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'food-7': {
            'translation': [3, 1.05, 5.6],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
    }
}

room_config_new = {
    'room': {
        'x_min': 0,
        'x_max': 6,
        'y_min': 0,
        'y_max': 3,
        'z_min': 0,
        'z_max': 6,
    },
    'objects': {
        'shelf-0': {
            'translation': [2, 0, 1],
            'scale': [1, 1, 1],
            'orientation': -90,
        },
        # 'shelf-1': {
        #     'translation': [2, 0, 3],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        'shelf-2': {
            'translation': [2, 0, 5],
            'scale': [1, 1, 1],
            'orientation': 90,
        },
        'wall_mounted_shelf-0': {
            'translation': [3, 1.5, 0],
            'scale': [1, 1, 1],
            'orientation': 180,
        },
        'wall_mounted_shelf-1': {
            'translation': [-2, 1.5, 3],
            'scale': [1, 1, 1],
            'orientation': -90,
        },
        'wall_mounted_shelf-2': {
            'translation': [3, 1.5, 6],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        'fridge_base': {
            'translation': [8, 0, 8],
            'scale': [1, 1, 1],
            'orientation': 180,
        },
        'worktop': {
            'translation': [6, 0, 1],
            'scale': [1, 1, 1],
            'orientation': 90,
        },
        'stove_top': {
            'translation': [5.65, 0.85, 0.7],
            'scale': [1, 1, 1],
            'orientation': 90,
        },
        'hood': {
            'translation': [6, 2, 0.7],
            'scale': [1, 1, 1],
            'orientation': 0,
        },
        # 'food-0': {
        #     'translation': [5, 0.39, 1],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-1': {
        #     'translation': [5, 0.74, 3],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-2': {
        #     'translation': [5, 1.09, 5],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-3': {
        #     'translation': [0.2, 1.27, 1],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-4': {
        #     'translation': [0.2, 1.83, 3],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-5': {
        #     'translation': [0.2, 2.15, 5],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-6': {
        #     'translation': [3, 0.6, 5.6],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
        # 'food-7': {
        #     'translation': [3, 1.05, 5.6],
        #     'scale': [1, 1, 1],
        #     'orientation': 0,
        # },
    },
    'ingredients': {
    }
}


def set_up_rerun(save_rrd=False):
    rr.init('Visualization: Kitchen', spawn=not save_rrd)
    if save_rrd:
        save_path = 'rerun-tmp'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        rr.save(os.path.join(save_path, 'Kitchen.rrd'))
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)


def write_scene(room_config=room_config_new):
    # write room
    box = room_config['room']
    room_min = np.array([box['x_min'], box['y_min'], box['z_min']])
    room_max = np.array([box['x_max'], box['y_max'], box['z_max']])
    rr.log('room', rr.Boxes3D(centers=[(room_min + room_max) / 2], sizes=[room_max - room_min], colors=[(0.5, 0.5, 0.5, 0.5)]))

    # write objects
    objects = room_config['objects']
    for obj_name, obj_config in objects.items():
        translation = obj_config['translation']
        orientation = obj_config['orientation']

        obj_path = os.path.join(config.SELECTED_ASSETS_PATH, f'{obj_name.split("-")[0]}.obj')
        object = trimesh.load(obj_path)
        vertices = object.vertices
        faces = object.faces

        center = object.bounding_box.centroid + translation  # 位移之后的中心点
        half_size = object.bounding_box.extents / 2  # extents 是边长不是半边长!!!
        if orientation == 90 or orientation == -90:
            half_size = half_size[[2, 1, 0]]  # 旋转之后的半边长

        offset = np.zeros(3)  # 初始化偏移数组为零
        for i in range(3):
            if center[i] - half_size[i] < room_min[i]:
                offset[i] = center[i] - half_size[i] - room_min[i]
            elif center[i] + half_size[i] > room_max[i]:
                offset[i] = center[i] + half_size[i] - room_max[i]

        # print(f'obj_name: {obj_name}, offset: {offset}, centroid: {object.bounding_box.centroid}, center: {center}, half_size: {half_size}')

        rotation = R.from_euler('xyz', [0, orientation, 0], degrees=True)
        vertices = rotation.apply(vertices)+translation-offset
        rr.log(f'objects/{obj_name}/mesh', rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
            vertex_normals=compute_vertex_normals(vertices, faces)))

        rr.log(f'objects/{obj_name}/bbox', rr.Transform3D(
            # scale=scale,
            translation=center - offset,
            rotation=rr.RotationAxisAngle(axis=(0, 1, 0), degrees=orientation),
        ))
        rr.log(f'objects/{obj_name}/bbox', rr.Boxes3D(centers=object.bounding_box.centroid, sizes=object.bounding_box.extents))


def write_human():
    # motion_path = os.path.join(config.DATA_HOLODECK_PATH, 'motion.pkl')
    # if not os.path.exists(motion_path):  # skip non-exist motion
    #     print(f'motion file not exist: {motion_path}')
    #     return
    # with open(motion_path, 'rb') as f:
    #     motion = pickle.load(f)

    # 生成一个简单的运动
    motion = {
        'translation': np.array([[1, 1.5, 3]]),
        'orientation': np.array([[0, 1.57, 0]]),
        'poses': np.zeros((1, 21, 3)),
    }

    frame_interval = 5
    fps = 30 / frame_interval
    motion_translation = motion['translation'][::frame_interval]
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


def main():
    set_up_rerun()
    write_scene()
    write_human()


if __name__ == '__main__':
    main()
