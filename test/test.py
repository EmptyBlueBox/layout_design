import json
import pickle
import numpy as np
import math
import os
import argparse
import sys
import open3d as o3d
import rerun as rr
from rerun.datatypes import Angle, RotationAxisAngle

np.set_printoptions(precision=6, suppress=True, linewidth=100, threshold=sys.maxsize)

NUM_EACH_CLASS = 4


def read_obj_vertices(obj_path):
    ''' This function reads the vertices from an .obj file.
    @Returns:
        vertices: N_vertices x 3
    '''
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f.readlines():
            if line.startswith('v '):
                vertices.append([float(a) for a in line.split()[1:]])
    return np.array(vertices)


def rotation_matrix(axis, theta):
    ''' This function computes the rotation matrix for a given axis and angle.
    @Returns:
        R: 3x3 rotation matrix
    '''
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])


def modelInfo2dict(model_info_path):
    ''' This function reads model information from a JSON file and returns it as a dictionary.
    @Returns:
        model_info_dict: dictionary with model_id as key and model info as value
    '''
    with open(model_info_path, 'r') as f:
        info = json.load(f)
    return {v['model_id']: v for v in info}


def gen_box_from_params(p):
    ''' This function generates a box from given parameters.
    @Returns:
        cornerpoints: 8x3 array of corner points of the box
    '''
    dir_1 = np.zeros((3))
    dir_1[:2] = p[:2]
    dir_1 = dir_1 / np.linalg.norm(dir_1)
    dir_2 = np.zeros((3))
    dir_2[:2] = [-dir_1[1], dir_1[0]]
    dir_3 = np.cross(dir_1, dir_2)

    center = p[3:6]
    size = p[6:9]

    cornerpoints = np.zeros([8, 3])
    d1 = 0.5 * size[1] * dir_1
    d2 = 0.5 * size[0] * dir_2
    d3 = 0.5 * size[2] * dir_3
    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3
    return cornerpoints


def get_transform(pos, rot, scale):
    ''' This function computes the transformation matrix for a given position, rotation, and scale.
    @Returns:
        transform: Transform3D object
    '''
    dref = [0, 0, 1]
    axis = np.cross(dref, rot)
    theta = np.arccos(np.dot(dref, rot)) * 2
    if np.sum(axis) != 0 and not math.isnan(theta):
        rotation = RotationAxisAngle(axis=axis, angle=Angle(rad=theta))
    else:
        rotation = RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=0))

    transform = rr.Transform3D(
        translation=pos,
        rotation=rotation,
        scale=scale
    )
    return transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--future_path', default='/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/3D-FUTURE-model', help='path to 3D FUTURE')
    parser.add_argument('--json_path', default='/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/3D-FRONT', help='path to 3D FRONT')
    parser.add_argument('--model_info_path', default='/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/model_info.json', help='path to model info')
    parser.add_argument('--save_path', default='./outputs', help='path to save result dir')
    parser.add_argument('--type', type=str, default='bedroom', help='bedroom or living')
    args = parser.parse_args()

    with open(f'./assets/cat2id_{args.type}.pkl', 'rb') as f:
        cat2id_dict = pickle.load(f)

    model_info_dict = modelInfo2dict(args.model_info_path)

    files = os.listdir(args.json_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.type == 'bedroom':
        room_types = ['Bedroom', 'MasterBedroom', 'SecondBedroom']
    if args.type == 'living':
        room_types = ['LivingDiningRoom', 'LivingRoom']
    layout_room_dict = {k: [] for k in room_types}
    test_room_dict = {k: None for k in room_types}  # 用于保存测试房间信息

    for n_m, m in enumerate(files):
        with open(os.path.join(args.json_path, m), 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_uid = [ff['uid'] for ff in data['furniture'] if 'valid' in ff and ff['valid']]
        model_jid = [ff['jid'] for ff in data['furniture'] if 'valid' in ff and ff['valid']]
        model_bbox = [ff['bbox'] for ff in data['furniture'] if 'valid' in ff and ff['valid']]

        mesh_uid = [mm['uid'] for mm in data['mesh']]
        mesh_xyz = [np.reshape(mm['xyz'], [-1, 3]) for mm in data['mesh']]
        mesh_faces = [np.reshape(mm['faces'], [-1, 3]) for mm in data['mesh']]

        scene = data['scene']
        room = scene['room']
        for r in room:
            if r['type'] not in room_types:
                continue

            layout = np.zeros((len(cat2id_dict) * NUM_EACH_CLASS, 10))
            layout_dict = {i: [] for i in range(len(cat2id_dict))}
            room_id = r['instanceid']
            children = r['children']

            for c in children:
                ref = c['ref']
                if ref not in model_uid:
                    continue
                idx = model_uid.index(ref)
                if not os.path.exists(os.path.join(args.future_path, model_jid[idx])):
                    continue

                v = read_obj_vertices(os.path.join(args.future_path, model_jid[idx], 'raw_model.obj'))
                center = (np.max(v, axis=0) + np.min(v, axis=0)) / 2
                hsize = (np.max(v, axis=0) - np.min(v, axis=0)) / 2  # half size
                bbox = center + np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1], [-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]) * hsize

                pos = c['pos']
                rot = c['rot'][1:]
                scale = c['scale']

                bbox = bbox * scale
                dref = [0, 0, 1]
                axis = np.cross(dref, rot)
                theta = np.arccos(np.dot(dref, rot)) * 2
                if np.sum(axis) != 0 and not math.isnan(theta):
                    R = rotation_matrix(axis, theta)
                    bbox = np.transpose(bbox)
                    bbox = np.matmul(R, bbox)
                    bbox = np.transpose(bbox)
                bbox = bbox + pos

                cn = np.mean(bbox, axis=0)
                dend = np.mean(bbox[4:, :], axis=0)
                di = dend - cn
                di = di / np.linalg.norm(di)
                sc = hsize * scale * 2

                bbox_me = gen_box_from_params(np.concatenate([[di[2], di[0], di[1]], [cn[2], cn[0], cn[1]], [sc[0], sc[2], sc[1]]]))
                cat_id = cat2id_dict.get(model_info_dict[model_jid[idx]]['category'], -1)
                if cat_id != -1:
                    layout_dict[cat_id].append(np.concatenate([[di[2], di[0], di[1]], [cn[2], cn[0], cn[1]], [sc[0], sc[2], sc[1]], [1]]))

            for k, v in layout_dict.items():
                if len(v) != 0:
                    lv = min(len(v), NUM_EACH_CLASS)
                    layout[k * NUM_EACH_CLASS: k * NUM_EACH_CLASS + lv] = np.stack(v[:lv], axis=0)

            layout_room_dict[r['type']].append(layout)
            if test_room_dict[r['type']] is None:
                test_room_dict[r['type']] = (layout, m, pos, rot, scale)  # 保存布局、文件名和变换信息

    for k, v in layout_room_dict.items():
        np.save(f'{args.save_path}/{k}.npy', np.stack(v, axis=0))

    for k, v in test_room_dict.items():
        if v is not None:
            layout, filename, pos, rot, scale = v
            np.save(f'{args.save_path}/{k}_test.npy', layout)

            # 使用 rerun 库保存 rrd 文件
            rr.init(f"object_with_mtl_{k}")
            rr.save(f"{args.save_path}/{k}_test.rrd")
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

            # 获取变换矩阵
            rr_transform = get_transform(pos, rot, scale)
            rr.log("world/asset", rr_transform)
            rr.log("world/asset", rr.Asset3D(path=os.path.join(args.future_path, filename)))
