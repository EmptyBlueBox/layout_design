import json
import categories
import os
import numpy as np
import rerun as rr
from tqdm import tqdm
import re
import config
from utils.mesh_utils import compute_vertex_normals

MIN_OBJ_NUM = 5 # 一个房间中最少的物体数量
save_path = f'./3DFRONT-preprocessed-MIN_OBJ_NUM_{MIN_OBJ_NUM}/'
SAVE_HOUSE = False


def get_model_id_2_info(info_path):
    ID_2_info = {}
    with open(info_path, 'r') as f:
        info = json.load(f)
    for object in info:
        ID_2_info[object['model_id']] = object
    return ID_2_info


def main():
    # 物体到物体信息的映射
    model_id_2_info = get_model_id_2_info(config.DATASET_3DFUTURE_MODEL_INFO_PATH)

    # 所有物体种类的名称列表
    catagory_name_2_super = {entey['category']: entey['super-category'] for entey in categories._CATEGORIES_3D}
    catagory_num = len(catagory_name_2_super)

    all_room_dict = {}  # 用于保存所有房间信息

    layouts = os.listdir(config.DATASET_3DFRONT_LAYOUT_PATH)
    layouts.sort()
    for layout in tqdm(layouts):
        layout_path = os.path.join(config.DATASET_3DFRONT_LAYOUT_PATH, layout)
        with open(layout_path, 'r', encoding='utf-8') as f:
            layout_info = json.load(f)

        # model_uid & model_jid store all furniture info of all rooms
        model_uid = []  # used to access 3D-FUTURE-model
        model_jid = []
        model_bbox = []
        for furniture in layout_info['furniture']:
            if 'valid' in furniture and furniture['valid']:
                model_uid.append(furniture['uid'])
                model_jid.append(furniture['jid'])
                model_bbox.append(furniture['bbox'])

        # mesh refers to wall/floor/etc
        mesh_uid = []
        mesh_vertices = []
        mesh_faces = []
        # print(f'mesh keys: {layout_info["mesh"][0].keys()}')
        # print(f'mesh aid: {layout_info["mesh"][0]["aid"]}')
        # print(f'mesh jid: {layout_info["mesh"][0]["jid"]}')
        # print(f'mesh type: {layout_info["mesh"][0]["type"]}')
        # print(f'mesh constructid: {layout_info["mesh"][0]["constructid"]}')
        if SAVE_HOUSE:
            rr.init(f'3DFRONT-layout-{layout}')
            rr.save(os.path.join(save_path, f'layout-{layout}.rrd'))
            rr.log('', rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        for mesh in layout_info['mesh']:
            mesh_uid.append(mesh['uid'])
            mesh_vertices.append(np.reshape(mesh['xyz'], [-1, 3]))
            mesh_faces.append(np.reshape(mesh['faces'], [-1, 3]))
            
            # print(f'mesh uid: {mesh["uid"]}')
            # print(f'    mesh aid: {mesh["aid"]}')
            # print(f'    mesh jid: {mesh["jid"]}')
            # print(f'    mesh type: {mesh["type"]}')
            # print(f'    mesh constructid: {mesh["constructid"]}')
            if SAVE_HOUSE:
                if mesh["type"] == 'Ceiling':
                    rr.log(f'Ceilings/{mesh_uid[-1]}', rr.Mesh3D(
                    vertex_positions=mesh_vertices[-1],
                    triangle_indices=mesh_faces[-1],
                    vertex_normals=compute_vertex_normals(mesh_vertices[-1], mesh_faces[-1])
                    ))
            
        # print(f'mesh num: {len(mesh_uid)}')
        scene = layout_info['scene']
        # print(f'scene keys: {scene.keys()}')
        rooms = scene['room']
        # print(f'room num: {len(rooms)}')
        # print(f'room keys: {rooms[0].keys()}')
        for room in rooms:
            # 如果房间类型不在指定的房间类型列表中，则跳过
            # print(f'room id: {room["instanceid"]}')
            # print(f'    room type: {room["type"]}')
            # print(f'    room size: {room["size"]}')
            # print(f'    room pos: {room["pos"]}')
            # print(f'    room rot: {room["rot"]}')
            # print(f'    room scale: {room["scale"]}')
            # print(f'    room empty: {room["empty"]}')
            # print(f'    room obj num: {len(room["children"])}')
            room_info = {'room_type': room['type'],  # 'Bedroom', 'MasterBedroom', 'SecondBedroom', ...
                         'room_id': room['instanceid'],  # 'Library-4425', not used
                         'object_list': []}  # list of object_info

            children = room['children']
            for child in children:
                ref = child['ref']
                if ref not in model_uid:
                    continue
                idx = model_uid.index(ref)
                if not os.path.exists(os.path.join(config.DATASET_3DFUTURE_MODEL_PATH, model_jid[idx])):
                    continue

                obj_catagory = model_id_2_info[model_jid[idx]]['category']
                if obj_catagory not in catagory_name_2_super and obj_catagory != None: # 出现数据错误就试图补全, 实际上确实会出现错误
                    for right_catagory_name in catagory_name_2_super:
                        if obj_catagory in right_catagory_name:
                            obj_catagory = right_catagory_name
                if obj_catagory not in catagory_name_2_super: # 如果补全后还是没有, 则跳过这个错误的物体名字
                    continue
                super_category = catagory_name_2_super[obj_catagory]
                object_info = {'obj_id': model_jid[idx],  # 3D-FUTURE-model id, can be used to access 3D-FUTURE .obj file
                               'super-category': super_category,  # 'Cabinet/Shelf/Desk', ...
                               'obj_catagory': obj_catagory,  # 'Children Cabinet', ...
                               'translation': np.array(child['pos']),  # [x, y, z]
                               'orientation': np.array(child['rot']),  # [x, y, z, w]
                               'scale': np.array(child['scale'])}  # [x, y, z]
                room_info['object_list'].append(object_info)
                
                if SAVE_HOUSE:
                    transform = rr.Transform3D(
                        translation=object_info['translation'],
                        rotation=rr.Quaternion(xyzw=object_info['orientation']),
                        scale=object_info['scale']
                    )
                    object_name = object_info['obj_catagory'].replace(' ', '')
                    rr.log(f'{room["instanceid"]}/{object_name}', transform)
                    rr.log(f'{room["instanceid"]}/{object_name}', rr.Asset3D(
                        path=os.path.join(config.DATASET_3DFUTURE_MODEL_PATH, object_info['obj_id'], 'raw_model.obj')
                    ))
                
            # 保存一个房间的信息, 物体太少就跳过
            if len(room_info['object_list']) < MIN_OBJ_NUM:
                continue
            if room_info['room_type'] in all_room_dict:
                all_room_dict[room_info['room_type']].append(room_info)
            else:
                all_room_dict[room_info['room_type']] = [room_info]

        if SAVE_HOUSE:
            print(f'house {layout} saved as 3DFRONT-layout-{layout}.rrd')
            exit(0) # 保存了一个房子, 退出

    # 保存所有房子的所有房间的信息, 先整理成 npy 数组, 再保存
    print(f'all room types: {all_room_dict.keys()}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 遍历所有房间类型
    for room_type, room_type_info in all_room_dict.items():
        sub_folder = os.path.join(save_path, room_type)
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)

        object_info_tobe_saved = {
            'obj_id': [],
            'super-category': [],
            'obj_catagory': [],
            'translation': [],
            'orientation': [],
            'scale': []
        }

        # 计算一类房间中物体的数量最大值, 便于后续整理成一样大小的 npy 数组
        max_obj_num = 0
        for room_info in room_type_info:
            max_obj_num = max(max_obj_num, len(room_info['object_list']))
        print(f'room type: {room_type}, room_num: {len(room_type_info)}, max_obj_num: {max_obj_num}')
        # 提取整理房间信息, 遍历每个房间
        for room_idx, room_info in enumerate(room_type_info):
            object_info_tobe_saved['obj_id'].append([''] * max_obj_num)
            object_info_tobe_saved['super-category'].append([''] * max_obj_num)
            object_info_tobe_saved['obj_catagory'].append([''] * max_obj_num)
            object_info_tobe_saved['translation'].append(np.zeros((max_obj_num, 3)))
            object_info_tobe_saved['orientation'].append(np.zeros((max_obj_num, 4)))
            object_info_tobe_saved['scale'].append(np.zeros((max_obj_num, 3)))
            # 遍历房间中的每个物体
            for object_idx, object_info in enumerate(room_info['object_list']):
                object_info_tobe_saved['obj_id'][room_idx][object_idx] = object_info['obj_id']
                object_info_tobe_saved['super-category'][room_idx][object_idx] = object_info['super-category']
                object_info_tobe_saved['obj_catagory'][room_idx][object_idx] = object_info['obj_catagory']
                object_info_tobe_saved['translation'][room_idx][object_idx] = object_info['translation']
                object_info_tobe_saved['orientation'][room_idx][object_idx] = object_info['orientation']
                object_info_tobe_saved['scale'][room_idx][object_idx] = object_info['scale']
        np.save(os.path.join(sub_folder, 'obj_id.npy'), np.array(object_info_tobe_saved['obj_id']))
        np.save(os.path.join(sub_folder, 'super-category.npy'), np.array(object_info_tobe_saved['super-category']))
        np.save(os.path.join(sub_folder, 'obj_catagory.npy'), np.array(object_info_tobe_saved['obj_catagory']))
        np.save(os.path.join(sub_folder, 'translation.npy'), np.array(object_info_tobe_saved['translation']))
        np.save(os.path.join(sub_folder, 'orientation.npy'), np.array(object_info_tobe_saved['orientation']))
        np.save(os.path.join(sub_folder, 'scale.npy'), np.array(object_info_tobe_saved['scale']))

        # 每种房间的第一个房间写入 Rerun, 用于可视化, 测试用
        room_name=room_type.replace(' ', '')
        rr.init(f'3DFRONT-visualization-{room_name}')
        rr.save(os.path.join(sub_folder, f'visualization-{room_name}.rrd'))
        rr.log('', rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        for object_idx in range(max_obj_num):
            if object_info_tobe_saved['obj_id'][0][object_idx] == '':
                break
            transform = rr.Transform3D(
                translation=object_info_tobe_saved['translation'][0][object_idx],
                rotation=rr.Quaternion(xyzw=object_info_tobe_saved['orientation'][0][object_idx]),
                scale=object_info_tobe_saved['scale'][0][object_idx]
            )
            object_name = object_info_tobe_saved['obj_catagory'][0][object_idx]
            object_name=object_name.replace(' ', '')
            rr.log(f'{object_name}', transform)
            rr.log(f'{object_name}', rr.Asset3D(
                path=os.path.join(config.DATASET_3DFUTURE_MODEL_PATH, object_info_tobe_saved['obj_id'][0][object_idx], 'raw_model.obj')
            ))


if __name__ == '__main__':
    main()
