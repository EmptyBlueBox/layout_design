import pickle
import gzip
import rerun as rr
import argparse
import numpy as np
import os
import config
import json
from utils.mesh_utils import compute_vertex_normals
from scipy.spatial.transform import Rotation as R
import smplx
import torch
from utils.mesh_utils import compute_vertex_normals

HOLODECK_NAME = 'a_DiningRoom_with_round_table_-2024-08-07-14-52-49-547177'
json_name = HOLODECK_NAME.split('-')[0]
save = True


def set_up_rerun():
    # start rerun script
    rr.init(f'Visualization: {HOLODECK_NAME}', spawn=not save)
    if save:
        rr.save(os.path.join('rerun', f'{json_name}.rrd'))
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)


def write_objects():
    # write objects
    holodeck_path = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, f'{json_name}.json')
    with open(holodeck_path, 'r') as f:
        holodeck_info = json.load(f)

    bb_min = np.array([np.inf, np.inf, np.inf])
    bb_max = np.array([-np.inf, -np.inf, -np.inf])
    for obj_info in holodeck_info['objects']:
        obj_name = obj_info['assetId']
        obj_id = obj_info['id']
        if '|' in obj_id:
            continue
        obj_path = os.path.join(config.OBJATHOR_BASE, obj_name, f'{obj_name}.pkl.gz')
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
               centers=(bb_min + bb_max - np.array([0, (bb_min[1]+bb_max[1])/2-1, 0])) / 2,
               half_sizes=[3, 1, 4],
               radii=0.01,
               colors=(255, 0, 0),
           ))


def write_human():
    # write human
    motion_translation_path = os.path.join(config.DATA_HOLODECK_PATH, 'test-motion', 'motion_translation.npy')
    motion_translation = np.load(motion_translation_path)
    motion_pose_path = os.path.join(config.DATA_HOLODECK_PATH, 'test-motion', 'motion_pose.npy')
    motion_pose = np.load(motion_pose_path).reshape(-1, 22, 3)

    assert len(motion_translation) == len(motion_pose)
    frame_interval = 5
    fps = 30 / frame_interval
    motion_translation = motion_translation[::frame_interval]
    motion_pose = motion_pose[::frame_interval]
    max_frame_num = motion_pose.shape[0]

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
    output = human_model(body_pose=torch.tensor(motion_pose[:, 1:], dtype=torch.float32),
                         global_orient=torch.tensor(motion_pose[:, 0], dtype=torch.float32),
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
    write_objects()
    write_human()


if __name__ == '__main__':
    main()
