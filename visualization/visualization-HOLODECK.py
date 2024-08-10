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

HOLODECK_NAME = 'a_DiningRoom_with_round_table_-2024-08-07-14-52-49-547177'
HOLODECK_NAME = 'a_warehouse_full_of_things-2024-08-07-14-41-48-293829'
HOLODECK_NAME = 'a_LivingDiningRoom_with_table,-2024-08-07-14-47-17-407752'

json_name = HOLODECK_NAME.split('-')[0]
holodeck_config = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, f'{json_name}.json')
with open(holodeck_config, 'r') as f:
    holodeck = json.load(f)

parser = argparse.ArgumentParser(description="Visualize HOLODECK data.")
rr.script_add_args(parser)
args = parser.parse_args()
rr.script_setup(args, f'Visualization: {HOLODECK_NAME}')
rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y

for obj_info in holodeck['objects']:
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
    triangles = np.array([[obj['triangles'][i], obj['triangles'][i+1], obj['triangles'][i+2]] for i in range(len(obj['triangles'])-3)])
    normals = compute_vertex_normals(vertices, triangles)

    rr.log(
        f'{obj_id.replace(" ", "-")}',
        rr.Mesh3D(
            vertex_positions=vertices,
            # triangle_indices=triangles,
            vertex_normals=normals,
        ),
    )

rr.script_teardown(args)
