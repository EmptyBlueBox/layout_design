'''
建构一个 Kitchen , 包含架子、冰箱、微波炉和工作台, 架子上面有箱子
使用 Rerun 可视化
并且保存三个文件, 包含物体点云、架子点云、物体与架子的关系, 输入给FLEX
对每一个架子/冰箱从 FLEX forward 一次
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
import json
import sys
sys.path.append('/Users/emptyblue/Documents/Research/FLEX/')


with open('/Users/emptyblue/Documents/Research/layout_design/dataset/SELECTED_ASSETS/kitchen.json', 'r') as f:
    room_config_new = json.load(f)

# shelf_name_save_for_FLEX = 'wall_mounted_shelf'  # 需要保存的架子名称, 同时也保存对应的物体, 物体架子关系
shelf_name_save_for_FLEX = 'shelf'  # 需要保存的架子名称, 同时也保存对应的物体, 物体架子关系
# shelf_name_save_for_FLEX = 'fridge_base'  # 需要保存的架子名称, 同时也保存对应的物体, 物体架子关系

HUMAN_PARAMS_ROOT = '/Users/emptyblue/Documents/Research/layout_design/dataset/SELECTED_ASSETS/save'  # 来自 FLEX 的人类参数


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

    # write fixtures
    fixtures = room_config['fixture']
    for fixture_name, obj_configs in fixtures.items():
        # 架子保存为 npz 格式, 给FLEX用的
        if fixture_name == shelf_name_save_for_FLEX:
            print(f'saving {shelf_name_save_for_FLEX} mesh for FLEX')

            obj_path = os.path.join(config.SELECTED_ASSETS_PATH, 'fixtures', f'{fixture_name}.obj')
            object_mesh = trimesh.load(obj_path)
            half_size = object_mesh.bounding_box.extents / 2  # extents 是边长不是半边长!!!
            fixture_vertices = object_mesh.vertices
            fixture_faces = object_mesh.faces

            R_y_2_z_up = R.from_euler('XYZ', [90, 0, 0], degrees=True)
            # 旋转平移之后的顶点
            z_up_vertices = R_y_2_z_up.apply(fixture_vertices) + np.array([0, 0, half_size[1]
                                                                           if fixture_name != 'wall_mounted_shelf' else obj_configs[0]['translation'][1]])
            receptacles = {shelf_name_save_for_FLEX: [[z_up_vertices, fixture_faces]]}
            receptacles_path = os.path.join(config.SELECTED_ASSETS_PATH, 'receptacles.npz')
            np.savez(receptacles_path, **receptacles)

        # write rerun
        for idx, obj_config in enumerate(obj_configs):
            obj_path = os.path.join(config.SELECTED_ASSETS_PATH, 'fixtures', f'{fixture_name}.obj')
            object_mesh = trimesh.load(obj_path)
            half_size = object_mesh.bounding_box.extents / 2  # extents 是边长不是半边长!!!
            fixture_vertices = object_mesh.vertices
            fixture_faces = object_mesh.faces

            fixture_translation = obj_config['translation']
            orientation = obj_config['orientation']

            center = object_mesh.bounding_box.centroid + fixture_translation  # 位移之后的中心点
            if orientation == 90 or orientation == -90:
                half_size = half_size[[2, 1, 0]]  # 旋转之后的半边长

            offset = np.zeros(3)  # 初始化偏移数组为零
            for i in range(3):
                if center[i] - half_size[i] < room_min[i]:
                    offset[i] = center[i] - half_size[i] - room_min[i]
                elif center[i] + half_size[i] > room_max[i]:
                    offset[i] = center[i] + half_size[i] - room_max[i]

            rotation = R.from_euler('xyz', angles=[0, orientation, 0], degrees=True)
            fixture_vertices = rotation.apply(fixture_vertices) + fixture_translation - offset

            # update json
            room_config['fixture'][fixture_name][idx]['translation'] = fixture_translation - offset

            # write rerun
            rr_path = f'fixtures/{fixture_name}/{idx}/'
            rr.log(rr_path+'mesh', rr.Mesh3D(
                vertex_positions=fixture_vertices,
                triangle_indices=fixture_faces,
                vertex_normals=compute_vertex_normals(fixture_vertices, fixture_faces)))

            rr.log(rr_path+'bbox', rr.Transform3D(
                # scale=scale,
                translation=center - offset,
                rotation=rr.RotationAxisAngle(axis=(0, 1, 0), degrees=orientation),
            ))
            rr.log(rr_path+'bbox', rr.Boxes3D(centers=object_mesh.bounding_box.centroid, sizes=object_mesh.bounding_box.extents))

    # # save updated room_config
    # with open('/Users/emptyblue/Documents/Research/layout_design/dataset/SELECTED_ASSETS/kitchen_updated.json', 'w') as f:
    #     json.dump(room_config, f, indent=4)

    # 物体与架子的关系, 需要保存为 npz 格式, 给FLEX用的
    dset_info = {}

    # human model
    human_model = smplx.create(model_path=config.SMPL_MODEL_PATH,
                               model_type='smplx',
                               gender='neutral',
                               num_pca_comps=np.array(24),
                               batch_size=1).to('cpu').eval()

    # write ingredients and human
    ingredients = room_config['ingredient']
    for obj_name, obj_configs in ingredients.items():
        for idx, obj_config in enumerate(obj_configs):  # idx: 这个物体出现的第几次
            fixture_name = obj_config['fixture']
            fixture_idx = obj_config['fixture_idx']  # 用第几个架子
            offset = obj_config['offset']
            fixture_path = os.path.join(config.SELECTED_ASSETS_PATH, 'fixtures', f'{fixture_name}.obj')
            fixture_mesh = trimesh.load(fixture_path)
            fixture_half_size = fixture_mesh.bounding_box.extents / 2
            fixture_y_rot = fixtures[fixture_name][fixture_idx]['orientation']

            obj_translation = np.array(fixtures[fixture_name][fixture_idx]['translation']) + np.array(offset)

            obj_path = os.path.join(config.SELECTED_ASSETS_PATH, 'ingredients', f'{obj_name}.obj')
            object_mesh = trimesh.load(obj_path)
            obj_vertices = object_mesh.vertices
            obj_faces = object_mesh.faces

            # 保存为 ply 格式, 给FLEX用的, 注意也要绕自己旋转
            ply_path = os.path.join(config.SELECTED_ASSETS_PATH, 'ingredients', 'ply_for_FLEX', f'{obj_name}.ply')
            R_y_2_z_up = R.from_euler('XYZ', [90, 0, 0], degrees=True)
            z_up_vertices = R_y_2_z_up.apply(obj_vertices)  # 旋转之后的顶点
            z_up_mesh = trimesh.Trimesh(vertices=z_up_vertices, faces=obj_faces)
            z_up_mesh.export(ply_path)

            # 保存相互关系, 生成数据给FLEX用的
            if fixture_name == shelf_name_save_for_FLEX:
                R_y_2_z_up = R.from_euler('XYZ', [90, 0, 0], degrees=True)
                offset = R_y_2_z_up.apply(offset) + np.array([0, 0, fixture_half_size[1]])  # 旋转之后的偏移
                holder_rotation_inv = R.from_euler('XYZ', angles=[0, 0, -fixtures[fixture_name][fixture_idx]['orientation']], degrees=True)
                dset_info[f'{obj_name}_{fixture_name}_all_{idx}'] = [offset, holder_rotation_inv.as_matrix(), 0]  # 物体偏移, 旋转矩阵, holder_idx=0 (唯一的架子)

            # 给物体上颜色, 红色的物体代表输出给 FLEX 的物体
            vertex_colors = np.zeros_like(obj_vertices)
            if fixture_name == shelf_name_save_for_FLEX:
                vertex_colors = np.ones_like(obj_vertices) * np.array([1, 0, 0])  # 红色
            else:
                vertex_colors = np.ones_like(obj_vertices) * np.array([0, 1, 0])  # 绿色

            # write objects rerun
            rr_path = f'ingredients/{obj_name}/{idx}/'
            rr.log(rr_path+'mesh', rr.Mesh3D(
                vertex_positions=obj_vertices + obj_translation,
                vertex_colors=vertex_colors,  # 颜色
                triangle_indices=obj_faces,
                vertex_normals=compute_vertex_normals(obj_vertices, obj_faces)))

            rr.log(rr_path+'bbox', rr.Boxes3D(centers=object_mesh.bounding_box.centroid + obj_translation, sizes=object_mesh.bounding_box.extents))

            # load human params
            human_params_path = os.path.join(HUMAN_PARAMS_ROOT, obj_name, fixture_name, f'all_{idx}.npz')
            if not os.path.exists(human_params_path):  # skip non-exist human params
                continue
            human_params = dict(np.load(human_params_path, allow_pickle=True))
            human_params = (human_params['arr_0'].item())['final_results']

            # write human rerun, 以物体作为参考点, 恢复到 y up 的坐标系
            for idx in range(5):
                res_i = human_params[idx]
                # print(f'idx: {idx}, res_i: {res_i.keys()}')

                # R_z_2_y_up1 = R.from_euler('xyz', angles=[0, -90, 0], degrees=True)
                # R_z_2_y_up2 = R.from_euler('xyz', [-90, -90, 0], degrees=True)
                # human_translation = R_z_2_y_up1.apply(res_i['transl'].reshape(
                #     1, 3)-np.array([0, 0, holder_half_size[1]])) + np.array(fixtures[fixture_name][fixture_idx]['translation'])
                # human_orientation = (R_z_2_y_up2*R.from_matrix(res_i['global_orient'])).as_rotvec().reshape(1, 3)
                human_translation = res_i['transl'].reshape(1, 3)
                human_orientation = (R.from_matrix(res_i['global_orient'])).as_rotvec().reshape(1, 3)
                human_pose = res_i['pose'].reshape(1, 63)

                output = human_model(body_pose=torch.tensor(human_pose, dtype=torch.float32),
                                     global_orient=torch.tensor(human_orientation, dtype=torch.float32),
                                     transl=torch.tensor(human_translation, dtype=torch.float32))

                vertices = output.vertices.detach().cpu().numpy()[0]
                R_z_2_y_up = R.from_euler('xyz', angles=[-90, fixture_y_rot, 0], degrees=True)
                vertices = R_z_2_y_up.apply(vertices - np.array([0, 0, fixture_half_size[1]])) + \
                    np.array(fixtures[fixture_name][fixture_idx]['translation'])
                faces = human_model.faces

                color = np.array([(4-idx)/4, 0, idx/4])  # 红色到蓝色的人
                # write human rerun
                rr.log(rr_path+f'human_{idx}',
                       rr.Mesh3D(vertex_positions=vertices,
                                 triangle_indices=faces,
                                 vertex_colors=np.ones_like(vertices)*color,
                                 vertex_normals=compute_vertex_normals(vertices, faces),))

    # 保存物体与架子的关系
    relationships_path = os.path.join(config.SELECTED_ASSETS_PATH, 'dset_info.npz')
    np.savez(relationships_path, **dset_info)


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
    # write_human()


if __name__ == '__main__':
    main()
