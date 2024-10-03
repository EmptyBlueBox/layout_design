"""
建构一个 Kitchen , 包含架子、冰箱、微波炉和工作台, 架子上面有箱子
使用 Rerun 可视化
并且保存三个文件, 包含物体点云、架子点云、物体与架子的关系, 输入给FLEX
对每一个架子/冰箱从 FLEX forward 一次
"""

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
from metric.metric_torque import get_torque
import copy
import cma

sys.path.append("/Users/emptyblue/Documents/Research/FLEX/")

# Hyperparameters
ROOM_RESOLUTION = 1  # 1m
weight_bbox = 100
weight_fatigue = 1
weight_distance = 1
weight_add = 100
path=['bread','bacon','onion','bread','steak','tomato','broccoli','bread']

# shelf_name_save_for_FLEX = 'wall_mounted_shelf'  # 需要保存的架子名称, 同时也保存对应的物体, 物体架子关系
shelf_name_save_for_FLEX = "shelf"  # 需要保存的架子名称, 同时也保存对应的物体, 物体架子关系
# shelf_name_save_for_FLEX = 'fridge_base'  # 需要保存的架子名称, 同时也保存对应的物体, 物体架子关系

with open(
    "/Users/emptyblue/Documents/Research/layout_design/dataset/SELECTED_ASSETS/kitchen.json",
    "r",
) as f:
    room_config_all = json.load(f)

# cache for bbox, fatigue, distance, 两次调用 loss_func 可以共用的信息
cache_bbox={} # bbox value, half size (3,)
cache_fatigue={} # max_torque value for legs and arms (1,)
cache_distance={} # x-z plane human translations (5, 2)

way_points=[]
motion_idx_chosen=[]

start_color=np.array([1.,0,0])
end_color=np.array([0,0,1.])
frame_time=1.
ingredient_color={
    'bread': np.array([234, 194, 131])/255,
    'bacon': np.array([220, 20, 60])/255,
    'onion': np.array([255, 240, 245])/255,
    'steak': np.array([139, 0, 0])/255,
    'tomato': np.array([1, 0, 0]),
    'broccoli': np.array([0, 128, 0])/255,
}

# human model
human_model = (
    smplx.create(
        model_path=config.SMPL_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        num_pca_comps=np.array(24),
        batch_size=1,
    )
    .to("cpu")
    .eval()
)

static_human_params_path = os.path.join(config.HUMAN_PARAMS_ROOT, 'tomato', 'shelf', "all_1.npz")
static_human_params = dict(np.load(static_human_params_path, allow_pickle=True))
static_human_params_poses = (static_human_params["arr_0"].item())["final_results"][0]["pose"].reshape(1, 63)
output = human_model(
    body_pose=torch.tensor(static_human_params_poses, dtype=torch.float32),
    global_orient=torch.tensor([[0,1.57,0]], dtype=torch.float32),
    transl=torch.tensor([[0,1.3,0]], dtype=torch.float32),
)
static_vertices = output.vertices.detach().cpu().numpy()[0]

def set_up_rerun(app_id='viz: kitchen',save_rrd=False):
    rr.init(app_id, spawn=not save_rrd)
    if save_rrd:
        save_path = "rerun-tmp"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        rr.save(os.path.join(save_path, "Kitchen.rrd"))
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)


def write_init_scene_human(room_config=room_config_all):
    # write room
    box = room_config["room"]
    room_min = np.array([box["x_min"], box["y_min"], box["z_min"]])
    room_max = np.array([box["x_max"], box["y_max"], box["z_max"]])
    rr.log(
        "room",
        rr.Boxes3D(
            centers=[(room_min + room_max) / 2],
            sizes=[room_max - room_min],
            colors=[(0.5, 0.5, 0.5, 0.5)],
        ),
    )

    # write fixtures
    fixtures = room_config["fixture"]
    for fixture_name, fixture_configs in fixtures.items():
        # 架子保存为 npz 格式, 给FLEX用的
        if fixture_name == shelf_name_save_for_FLEX:
            print(f"saving {shelf_name_save_for_FLEX} mesh for FLEX")

            obj_path = os.path.join(
                config.SELECTED_ASSETS_PATH, "fixtures", f"{fixture_name}.obj"
            )
            object_mesh = trimesh.load(obj_path)
            half_size = object_mesh.bounding_box.extents / 2 # extents 是边长不是半边长!!!
            fixture_vertices = object_mesh.vertices
            fixture_faces = object_mesh.faces

            R_y_2_z_up = R.from_euler("XYZ", [90, 0, 0], degrees=True) # 旋转平移之后的顶点
            z_up_vertices = R_y_2_z_up.apply(fixture_vertices) + np.array(
                [
                    0,
                    0,
                    half_size[1]
                    if fixture_name != "wall_mounted_shelf"
                    else fixture_configs[0]["translation"][1],
                ]
            )
            receptacles = {shelf_name_save_for_FLEX: [[z_up_vertices, fixture_faces]]}
            receptacles_path = os.path.join(
                config.SELECTED_ASSETS_PATH, "receptacles.npz"
            )
            np.savez(receptacles_path, **receptacles)

        # write rerun
        for idx, fixture_config in enumerate(fixture_configs):
            obj_path = os.path.join(
                config.SELECTED_ASSETS_PATH, "fixtures", f"{fixture_name}.obj"
            )
            object_mesh = trimesh.load(obj_path)
            half_size = object_mesh.bounding_box.extents / 2 # extents 是边长不是半边长!!!
            fixture_vertices = object_mesh.vertices
            fixture_faces = object_mesh.faces

            fixture_translation = fixture_config["translation"]
            orientation = fixture_config["orientation"]

            center = object_mesh.bounding_box.centroid + fixture_translation # 位移之后的中心点
            if orientation == 90 or orientation == -90:
                half_size = half_size[[2, 1, 0]]  # 旋转之后的半边长

            offset = np.zeros(3)  # 初始化偏移数组为零
            for i in range(3):
                if center[i] - half_size[i] < room_min[i]:
                    offset[i] = center[i] - half_size[i] - room_min[i]
                elif center[i] + half_size[i] > room_max[i]:
                    offset[i] = center[i] + half_size[i] - room_max[i]

            rotation = R.from_euler("xyz", angles=[0, orientation, 0], degrees=True)
            fixture_vertices = (
                rotation.apply(fixture_vertices) + fixture_translation - offset
            )

            # update json
            room_config["fixture"][fixture_name][idx]["translation"] = (
                fixture_translation - offset
            )

            # write rerun
            rr_path = f"fixtures/{fixture_name}/{idx}/"
            rr.log(
                rr_path + "mesh",
                rr.Mesh3D(
                    vertex_positions=fixture_vertices,
                    triangle_indices=fixture_faces,
                    vertex_normals=compute_vertex_normals(
                        fixture_vertices, fixture_faces
                    ),
                ),
            )

            # rr.log(
            #     rr_path + "bbox",
            #     rr.Transform3D(
            #         # scale=scale,
            #         translation=center - offset,
            #         rotation=rr.RotationAxisAngle(axis=(0, 1, 0), degrees=orientation),
            #     ),
            # )
            # rr.log(
            #     rr_path + "bbox",
            #     rr.Boxes3D(
            #         centers=object_mesh.bounding_box.centroid,
            #         sizes=object_mesh.bounding_box.extents,
            #     ),
            # )

    # 物体与架子的关系, 需要保存为 npz 格式, 给FLEX用的
    dset_info = {}

    # write ingredients and human
    ingredients = room_config["ingredient"]
    for obj_name, obj_configs in ingredients.items():
        for idx, obj_config in enumerate(obj_configs):  # idx: 这个物体出现的第几次
            fixture_name = obj_config["fixture"]
            fixture_idx = obj_config["fixture_idx"]  # 用第几个架子
            fixture_path = os.path.join(
                config.SELECTED_ASSETS_PATH, "fixtures", f"{fixture_name}.obj"
            )
            fixture_mesh = trimesh.load(fixture_path)
            fixture_half_size = fixture_mesh.bounding_box.extents / 2
            fixture_y_rot = fixtures[fixture_name][fixture_idx]["orientation"]

            offset = obj_config["offset"]
            obj_translation = np.array(
                fixtures[fixture_name][fixture_idx]["translation"]
            ) + np.array(offset)

            obj_path = os.path.join(
                config.SELECTED_ASSETS_PATH, "ingredients", f"{obj_name}.obj"
            )
            object_mesh = trimesh.load(obj_path)
            obj_vertices = object_mesh.vertices
            obj_faces = object_mesh.faces

            # 保存为 ply 格式, 给FLEX用的, 注意也要绕自己旋转
            ply_path = os.path.join(
                config.SELECTED_ASSETS_PATH,
                "ingredients",
                "ply_for_FLEX",
                f"{obj_name}.ply",
            )
            R_y_2_z_up = R.from_euler("XYZ", [90, 0, 0], degrees=True)
            z_up_vertices = R_y_2_z_up.apply(obj_vertices)  # 旋转之后的顶点
            z_up_mesh = trimesh.Trimesh(vertices=z_up_vertices, faces=obj_faces)
            z_up_mesh.export(ply_path)

            # 保存相互关系, 生成数据给FLEX用的
            if fixture_name == shelf_name_save_for_FLEX:
                R_y_2_z_up = R.from_euler("XYZ", [90, 0, 0], degrees=True)
                offset = R_y_2_z_up.apply(offset) + np.array(
                    [0, 0, fixture_half_size[1] if fixture_name != "wall_mounted_shelf" else room_config["fixture"][fixture_name][0]["translation"][1]]
                )  # 旋转之后的偏移
                holder_rotation_inv = R.from_euler(
                    "XYZ",
                    angles=[0, 0, -fixtures[fixture_name][fixture_idx]["orientation"]],
                    degrees=True,
                )
                dset_info[f"{obj_name}_{fixture_name}_all_{idx}"] = [
                    offset,
                    holder_rotation_inv.as_matrix(),
                    0,
                ]  # 物体偏移, 旋转矩阵, holder_idx=0 (唯一的架子)

            # 给物体上颜色, 红色的物体代表输出给 FLEX 的物体
            vertex_colors = np.zeros_like(obj_vertices)
            if fixture_name == shelf_name_save_for_FLEX:
                vertex_colors = np.ones_like(obj_vertices) * np.array([1, 0, 0])  # 红色
            else:
                vertex_colors = np.ones_like(obj_vertices) * np.array([0, 1, 0])  # 绿色

            # write objects rerun
            vertex_colors=np.ones_like(obj_vertices)*ingredient_color[obj_name]
            rr.set_time_seconds("stable_time", 0)
            rr_path = f"ingredients/{obj_name}/{idx}/"
            rr.log(
                rr_path + "mesh",
                rr.Mesh3D(
                    vertex_positions=obj_vertices + obj_translation,
                    vertex_colors=vertex_colors,  # 颜色
                    triangle_indices=obj_faces,
                    vertex_normals=compute_vertex_normals(obj_vertices, obj_faces),
                ),
            )

            # rr.log(
            #     rr_path + "bbox",
            #     rr.Boxes3D(
            #         centers=object_mesh.bounding_box.centroid + obj_translation,
            #         sizes=object_mesh.bounding_box.extents,
            #     ),
            # )

            # load human params
            if 'idx_for_viz' in obj_config:
                idx_for_viz=obj_config['idx_for_viz']
                human_params_path = os.path.join(
                    config.HUMAN_PARAMS_ROOT, obj_name, fixture_name, f"all_{idx_for_viz}.npz"
                )
            else:
                human_params_path = os.path.join(
                    config.HUMAN_PARAMS_ROOT, obj_name, fixture_name, f"all_{idx}.npz"
                )
                
            if not os.path.exists(human_params_path):  # skip non-exist human params
                continue
            human_params = dict(np.load(human_params_path, allow_pickle=True))
            human_params = (human_params["arr_0"].item())["final_results"]

            # write human rerun, 以物体作为参考点, 恢复到 y up 的坐标系
            # print(way_points, motion_idx_chosen)
            for idx in (range(5) if way_points == [] else range(len(way_points))):
                # print(111, idx)
                if way_points != [] and path[idx] != obj_name:
                    continue
                elif way_points != [] and path[idx] == obj_name:
                    # print(222, idx)
                    color_idx=idx
                    idx=motion_idx_chosen[idx]
                res_i = human_params[idx]
                # print(f'idx: {idx}, res_i: {res_i.keys()}')
                R_z_2_y_up = R.from_euler(
                    "xyz", angles=[-90, fixture_y_rot, 0], degrees=True
                )
                
                fixture_translation=fixtures[fixture_name][fixture_idx]["translation"]
                human_translation = res_i["transl"].reshape(1, 3) - (np.array([[0, 0, fixture_half_size[1]]]) if fixture_name == "wall_mounted_shelf" else np.zeros((1,3)))
                
                #防止越过边界
                tmp_transl=R_z_2_y_up.apply(human_translation)
                if fixture_translation[0]+tmp_transl[0,0]<room_min[0] or fixture_translation[0]+tmp_transl[0,0]>room_max[0]:
                    continue
                # if fixture_translation[2]+tmp_transl[0,2]<room_min[2] or fixture_translation[2]+tmp_transl[0,2]>room_max[2]:
                #     continue
                # COMMENT
                # print(f'{fixture_name}: {fixture_translation}, human_translation: {human_translation}')
                
                # 计算人体方向
                human_orientation = (
                    (R.from_matrix(res_i["global_orient"])).as_rotvec().reshape(1, 3)
                )
                human_pose = res_i["pose"].reshape(1, 63)

                output = human_model(
                    body_pose=torch.tensor(human_pose, dtype=torch.float32),
                    global_orient=torch.tensor(human_orientation, dtype=torch.float32),
                    transl=torch.tensor(human_translation, dtype=torch.float32),
                )

                vertices = output.vertices.detach().cpu().numpy()[0]

                vertices = R_z_2_y_up.apply(
                    vertices - np.array([0, 0, fixture_half_size[1]])
                ) + np.array(fixture_translation)
                faces = human_model.faces

                if way_points == []:# 红色到蓝色的人
                    color=start_color*(4-idx)/4+end_color*idx/4
                else:
                    # print(f'color_idx: {color_idx}')
                    color=start_color*(7-color_idx)/7+end_color*color_idx/7
                # write human rerun
                if way_points != []:
                    rr.set_time_seconds("stable_time", color_idx*frame_time)
                    rr.log(
                    rr_path + f"human_{idx}",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=np.ones_like(vertices) * color,
                        vertex_normals=compute_vertex_normals(vertices, faces),
                    ),
                    )                    
                    # print('test11')
                    rr.set_time_seconds("stable_time", (color_idx+2)*frame_time)
                    rr.log(
                    rr_path + f"human_{idx}",
                    rr.Mesh3D(
                        vertex_positions=np.zeros_like(vertices),
                        triangle_indices=faces,
                    ),
                    ) 
                    rr.set_time_seconds("stable_time", 0)
                    rr.log(
                    rr_path + f"human_{idx}",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=np.ones_like(vertices) * color,
                        vertex_normals=compute_vertex_normals(vertices, faces),
                    ),
                    )  
                    # print('test22')
                    if color_idx==0:# 给初始地方也加一个人
                        rr.set_time_seconds("stable_time", 0)
                        tmp_static_vertices=static_vertices+np.array([5, 0, 3])
                        rr.log(
                        rr_path + f"human0_{idx}",
                        rr.Mesh3D(
                            vertex_positions=tmp_static_vertices,
                            triangle_indices=faces,
                            vertex_colors=np.ones_like(tmp_static_vertices) * color,
                            vertex_normals=compute_vertex_normals(tmp_static_vertices, faces),
                        ),)
                        rr.set_time_seconds("stable_time", frame_time)
                        rr.log(
                        rr_path + f"human0_{idx}",
                        rr.Mesh3D(
                            vertex_positions=np.zeros_like(vertices),
                            triangle_indices=faces,
                        ),
                        )
                        rr.set_time_seconds("stable_time", 0)
                    if color_idx==len(way_points)-1: # 给最后地方也加2个人
                        rr.set_time_seconds("stable_time", color_idx*frame_time)
                        tmp_static_vertices=static_vertices+np.array([5, 0, 3])
                        rr.log(
                        rr_path + f"human_last_worktop_{idx}",
                        rr.Mesh3D(
                            vertex_positions=tmp_static_vertices,
                            triangle_indices=faces,
                            vertex_colors=np.ones_like(tmp_static_vertices) * color,
                            vertex_normals=compute_vertex_normals(tmp_static_vertices, faces),
                        ),)
                        rr.set_time_seconds("stable_time", (color_idx+1)*frame_time)
                        rr.log(
                        rr_path + f"human_last_worktop_{idx}",
                        rr.Mesh3D(
                            vertex_positions=np.zeros_like(vertices),
                            triangle_indices=faces,
                        ),
                        )
                        
                        rr.set_time_seconds("stable_time", color_idx*frame_time)
                        tmp_R=R.from_euler('xyz', [0, 180, 0], degrees=True)
                        tmp_static_vertices=tmp_R.apply(static_vertices)+np.array(room_config_all['fixture']['table'][0]['translation'])+np.array([0.7,0,0])
                        rr.log(
                        rr_path + f"human_last_table_{idx}",
                        rr.Mesh3D(
                            vertex_positions=tmp_static_vertices,
                            triangle_indices=faces,
                            vertex_colors=np.ones_like(tmp_static_vertices) * color,
                            vertex_normals=compute_vertex_normals(tmp_static_vertices, faces),
                        ),)
                        rr.set_time_seconds("stable_time", (color_idx+1)*frame_time)
                        rr.log(
                        rr_path + f"human_last_table_{idx}",
                        rr.Mesh3D(
                            vertex_positions=np.zeros_like(vertices),
                            triangle_indices=faces,
                        ),
                        )
                else:
                    rr.log(
                        rr_path + f"human_{idx}",
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            # vertex_colors=np.ones_like(vertices) * color,
                            vertex_normals=compute_vertex_normals(vertices, faces),
                        ),
                    )
                    
                idx=color_idx

    # 保存物体与架子的关系
    relationships_path = os.path.join(config.SELECTED_ASSETS_PATH, "dset_info.npz")
    np.savez(relationships_path, **dset_info)


def bbox_loss(room_config, viz=False):
    fixture_config_all = room_config["fixture"]
    bboxs_halfsize = []
    bboxs_translation = []
    bboxs_orientation = []
    
    for fixture_name, fixture_configs in fixture_config_all.items():
        for idx, fixture_config in enumerate(fixture_configs):            
            # load fixture bbox from cache
            cache_name = f"{fixture_name}"
            if cache_name in cache_bbox:
                bbox_half_size = cache_bbox[cache_name]  # (3,)
            else:
                obj_path = os.path.join(config.SELECTED_ASSETS_PATH, "fixtures", f"{fixture_name}.obj")
                object_mesh = trimesh.load(obj_path)
                bbox_half_size = object_mesh.bounding_box.extents / 2  # (3,)
                cache_bbox[cache_name] = bbox_half_size  # 更新缓存
            
            # get fixture translation
            bbox_translation = fixture_config["translation"]  # (3,)
            bbox_orientation = fixture_config["orientation"]  # (1,), degree
            
            # update tmp cache
            if bbox_orientation == 90 or bbox_orientation == -90:
                bbox_half_size = bbox_half_size[[2, 1, 0]]  # (3,)
            bboxs_halfsize.append(bbox_half_size)
            bboxs_translation.append(bbox_translation)
            bboxs_orientation.append(bbox_orientation)
    
    def compute_intersection_depth(bbox1, bbox2):
        """计算两个轴对齐的边界框的嵌入深度"""
        t1, h1 = bbox1
        t2, h2 = bbox2
        
        # 计算在各个轴上的重叠量
        overlap_x = max(0, min(t1[0] + h1[0], t2[0] + h2[0]) - max(t1[0] - h1[0], t2[0] - h2[0]))
        overlap_y = max(0, min(t1[1] + h1[1], t2[1] + h2[1]) - max(t1[1] - h1[1], t2[1] - h2[1]))
        overlap_z = max(0, min(t1[2] + h1[2], t2[2] + h2[2]) - max(t1[2] - h1[2], t2[2] - h2[2]))
        
        # 返回嵌入深度
        return min(overlap_x, overlap_y, overlap_z)
    
    max_intersection_depth = 0
    
    # 遍历所有的 bbox 对，计算相交的最大嵌入深度
    num_bboxes = len(bboxs_halfsize)
    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            bbox1 = (bboxs_translation[i], bboxs_halfsize[i])
            bbox2 = (bboxs_translation[j], bboxs_halfsize[j])
            
            intersection_depth = compute_intersection_depth(bbox1, bbox2)
            if intersection_depth > max_intersection_depth:
                max_intersection_depth = intersection_depth
    
    return max_intersection_depth


def fatigue_loss(room_config, viz=False):
    torque_sum=0
    
    # print(f'all ingredients: {room_config["ingredient"]}')
    
    ingredient_configs = room_config["ingredient"]
    for ingredient_name, ingredient_configs in ingredient_configs.items():
        # print(f'fatigue loss for {ingredient_name}')
        for idx, human_config in enumerate(ingredient_configs):
            
            viz_idx=human_config['idx_for_viz'] if 'idx_for_viz' in human_config else idx
            cache_name=f"{ingredient_name}_{viz_idx}"
            
            if cache_name in cache_fatigue:
                max_torque=cache_fatigue[cache_name]
            else:
                fixture_name = human_config["fixture"]
                human_params_path = os.path.join(
                    config.HUMAN_PARAMS_ROOT, ingredient_name, fixture_name, f"all_{viz_idx}.npz"
                )
                if not os.path.exists(human_params_path):  # skip non-exist human params
                    continue
                human_params = dict(np.load(human_params_path, allow_pickle=True))
                human_params = (human_params["arr_0"].item())["final_results"]
                
                fixture_half_size = cache_bbox[fixture_name]
                human_translation = human_params[0]["transl"].reshape(1, 3) - (np.array([[0, 0, fixture_half_size[1]]]) if fixture_name == "wall_mounted_shelf" else np.zeros((1,3)))
                R_z_2_y_up2 = R.from_euler('xyz', [-90, -90, 0], degrees=True)
                human_orientation = (R_z_2_y_up2*(R.from_matrix(human_params[0]["global_orient"]))).as_rotvec().reshape(1, 3) # 转换到y up坐标系
                human_pose = human_params[0]["pose"].reshape(1, 21, 3)
                
                human_params = {'translation': human_translation, 'orientation': human_orientation, 'poses': human_pose}
                est_torque = get_torque(human_params).reshape(23,3)
                
                #15,16左臂, 19,20右臂, 2左腿, 6右腿
                all_candidates_idx = [15,16,19,20,2,6]
                all_candidates_torque = np.linalg.norm(est_torque[all_candidates_idx], axis=1)
                max_torque = np.max(all_candidates_torque)
                # print(f'{cache_name}: {max_torque}')
                # 更新缓存
                cache_fatigue[cache_name]=max_torque
            
            print(f'{cache_name}: {max_torque}')
            torque_sum += max_torque
    
    mean_fatigue = torque_sum / len(ingredient_configs)      
    return mean_fatigue    


def distance_loss(room_config, viz=False):
    fixture_config_all = room_config["fixture"]
    bboxs_halfsize={} # {架子名+id: (2,)} x,z plane
    bboxs_translation={} # {架子名+id: (2,)} x,z plane
    bboxs_orientation={} # {架子名+id: (1,)} degree
    for fixture_name, fixture_configs in fixture_config_all.items():
        for idx, fixture_config in enumerate(fixture_configs):            
            # load fixture bbox from cache
            cache_name=f"{fixture_name}_{idx}"
            # print(f'{cache_name} in cache_bbox: {cache_name in cache_bbox}')
            if cache_name in cache_bbox:
                bbox_half_size=cache_bbox[cache_name] # (3,)
            else:
                obj_path = os.path.join(config.SELECTED_ASSETS_PATH, "fixtures", f"{fixture_name}.obj")
                object_mesh = trimesh.load(obj_path)
                bbox_half_size = object_mesh.bounding_box.extents / 2 #(3,)
                cache_bbox[cache_name]=bbox_half_size # 更新缓存
            
            # get fixture translation
            bbox_translation = fixture_config["translation"] #(3,)
            bbox_orientation = fixture_config["orientation"] # (1,), degree
            
            bboxs_halfsize[cache_name]=bbox_half_size[[0,2]]
            bboxs_translation[cache_name]=bbox_translation[[0,2]]
            bboxs_orientation[cache_name]=bbox_orientation
            
    
    shelf_has_ingredient={} # {架子名+id: [物体名]}
    ingredient_in_shelf={} # {物体名: [架子名+id]}
    shelf_ingredients_direction={} # {架子名+id+物体名: 5*[x,z]}
    
    ingredient_configs = room_config["ingredient"]
    for ingredient_name, ingredient_configs in ingredient_configs.items():
        for idx, ingredient_config in enumerate(ingredient_configs):
            
            fixture_name = ingredient_config["fixture"]
            fixture_idx = ingredient_config["fixture_idx"]
            fixture_hash=f"{fixture_name}_{fixture_idx}"
            shelf_has_ingredient[fixture_hash]=[ingredient_name] if fixture_hash not in ingredient_in_shelf else ingredient_in_shelf[fixture_hash]+[ingredient_name]
            ingredient_in_shelf[ingredient_name]=[fixture_hash] if ingredient_name not in ingredient_in_shelf else ingredient_in_shelf[ingredient_name]+[fixture_hash]
            
            if 'idx_for_viz' in ingredient_config:
                idx=ingredient_config['idx_for_viz']
            cache_name=f"{ingredient_name}_{idx}"
            if cache_name in cache_distance:
                all_human_translation=cache_distance[cache_name]
                # print(cache_name,1)
            else:
                # print(cache_name,2)
                all_human_translation = []
                for motion_idx in range(5):
                    human_params_path = os.path.join(
                        config.HUMAN_PARAMS_ROOT, ingredient_name, fixture_name, f"all_{idx}.npz"
                    )
                    if not os.path.exists(human_params_path):  # skip non-exist human params
                        print(f'{human_params_path} not exist')
                        continue
                    human_params = dict(np.load(human_params_path, allow_pickle=True))
                    human_params = (human_params["arr_0"].item())["final_results"]
                    
                    human_translation = human_params[motion_idx]["transl"].reshape(-1)
                    R_z_2_y_up = R.from_euler("xyz", angles=[-90, bboxs_orientation[fixture_hash], 0], degrees=True)
                    tmp_translation = R_z_2_y_up.apply(human_translation)
                    all_human_translation.append(tmp_translation[[0,2]])
                all_human_translation=np.array(all_human_translation)
                
                cache_distance[cache_name]=all_human_translation# 更新缓存
            
            shelf_ingredients_direction[f'{fixture_hash}_{ingredient_name}']=all_human_translation
            # print(f"{fixture_hash}_{ingredient_name}: {all_human_translation.shape}")
            
    # print(f'shelf_ingredients_direction: {shelf_ingredients_direction.keys()}')
    # 贪心计算最短路径
    now=np.array([5,3])
    global way_points
    way_points=[]
    global motion_idx_chosen
    motion_idx_chosen=[]
    
    total_distance=0
    prev_shelf_ingredients_direction=None
    for need_ingredient in path:
        potential_shelf=ingredient_in_shelf[need_ingredient]
        min_distance = np.inf
        next_shelf_hash=None
        vector_to_shelf=None #上一个架子到下一个架子的方向
        
        # 选择最近的有下一个物品的架子, shelf+id
        for shelf_hash in potential_shelf: 
            dis=np.linalg.norm(bboxs_translation[shelf_hash]-now)
            if dis<min_distance:
                min_distance=dis
                next_shelf_hash=shelf_hash
                vector_to_shelf=bboxs_translation[shelf_hash]-now
        
        if viz:
            time_idx=len(motion_idx_chosen)
            rr.set_time_seconds("stable_time", time_idx*frame_time)
            rr.log(
                f"path/{need_ingredient}",
                rr.Arrows3D(
                    origins=[now[0], 1, now[1]],
                    vectors=[vector_to_shelf[0]/2, 0, vector_to_shelf[1]/2],
                    colors=start_color*(7-time_idx)/7+end_color*time_idx/7,
                    radii=0.1,
                ),
            )
            rr.set_time_seconds("stable_time", (time_idx+1)*frame_time)
            rr.log(
                f"path/{need_ingredient}",
                rr.Arrows3D(
                    origins=[now[0], 0.5, now[1]],
                    vectors=[0,0,0],
                ),
            )
            rr.set_time_seconds(timeline="stable_time", seconds=0)
            if time_idx == 7: # 画出回到 [5,3] 和 np.array(room_config_all['fixture']['table'][0]['translation'])-np.array([1,0,0]) 的箭头
                rr.set_time_seconds("stable_time", 7*frame_time)
                rr.log(
                    f"path/top_{need_ingredient}",
                    rr.Arrows3D(
                        origins=[now[0]+vector_to_shelf[0], 1, now[1]+vector_to_shelf[1]],
                        vectors=np.array([5-now[0]-vector_to_shelf[0], 0, 3-now[1]-vector_to_shelf[1]])*2/3,
                        colors=start_color*(7-time_idx)/7+end_color*time_idx/7,
                        radii=0.1,
                    ),
                )
                rr.log(
                    f"path/table_{need_ingredient}",
                    rr.Arrows3D(
                        origins=[5, 1, 3],
                        vectors=(np.array(room_config_all['fixture']['table'][0]['translation'])+np.array([1,0,0])-np.array([5,0,3]))*2/3,
                        colors=start_color*(7-time_idx)/7+end_color*time_idx/7,
                        radii=0.1,
                    ),
                )
                rr.set_time_seconds("stable_time", 8*frame_time)
                rr.log(
                    f"path/top_{need_ingredient}",
                    rr.Arrows3D(
                        origins=[5, 1, 3],
                        vectors=[0,0,0],
                    ),
                )
                rr.log(
                    f"path/table_{need_ingredient}",
                    rr.Arrows3D(
                        origins=[5, 1, 3],
                        vectors=[0,0,0],
                    ),
                )
                rr.set_time_seconds(timeline="stable_time", seconds=0)
        # print(f'{next_shelf_hash}_{need_ingredient}')
        # 下一个架子的所有motion方向
        candidate_ingredients=np.array(shelf_ingredients_direction[f'{next_shelf_hash}_{need_ingredient}'])
        # 看看每一个motion哪一个和来时候的方向最接近
        try_all_motion = np.dot(np.tile(vector_to_shelf, (5, 1)), candidate_ingredients.T)
        min_idx=np.argmin(try_all_motion)
        #更新当前位置
        now=bboxs_translation[next_shelf_hash]+candidate_ingredients[min_idx]
        if now[0]<0 or now[0]>6 or now[1]<0 or now[1]>6:
            idx_mask=np.ones(5, dtype=bool)
            idx_mask[min_idx]=False
            min_idx=np.argmin(try_all_motion[idx_mask])
        now=bboxs_translation[next_shelf_hash]+candidate_ingredients[min_idx]
        if now[0]<0 or now[0]>6 or now[1]<0 or now[1]>6:
            idx_mask=np.ones(5, dtype=bool)
            idx_mask[min_idx]=False
            min_idx=np.argmin(try_all_motion[idx_mask])
        now=bboxs_translation[next_shelf_hash]+candidate_ingredients[min_idx]
        if now[0]<0 or now[0]>6 or now[1]<0 or now[1]>6:
            idx_mask=np.ones(5, dtype=bool)
            idx_mask[min_idx]=False
            min_idx=np.argmin(try_all_motion[idx_mask])
        now=bboxs_translation[next_shelf_hash]+candidate_ingredients[min_idx]
        if now[0]<0 or now[0]>6 or now[1]<0 or now[1]>6:
            idx_mask=np.ones(5, dtype=bool)
            idx_mask[min_idx]=False
            min_idx=np.argmin(try_all_motion[idx_mask])
        now=bboxs_translation[next_shelf_hash]+candidate_ingredients[min_idx]
        # 保存选取的motion
        motion_idx_chosen.append(min_idx)

        # 选择最近的架子, shelf+id 放到路径中
        way_points.append(next_shelf_hash)
        # 更新总距离
        total_distance+=min_distance
        # 穿过了目标架子
        if np.dot(vector_to_shelf, candidate_ingredients[min_idx])>0: 
            total_distance+=np.sum(bboxs_halfsize[next_shelf_hash])/2
        # 穿过了上一个架子, 利用保存的上一个motion相对上一个架子的方向
        if prev_shelf_ingredients_direction is not None:
            if np.dot(vector_to_shelf, prev_shelf_ingredients_direction)<0:
                total_distance+=np.sum(bboxs_halfsize[next_shelf_hash])/2
        # 更新上一个架子的方向
        prev_shelf_ingredients_direction=candidate_ingredients[min_idx]
    
    return total_distance


def loss_func(room_config, viz=False):
    box = room_config["room"]
    room_min = np.array([box["x_min"], box["y_min"], box["z_min"]])
    room_max = np.array([box["x_max"], box["y_max"], box["z_max"]])
    
    # # write rerun room
    # if viz:
    #     set_up_rerun(app_id=f"viz: kitchen_{epoch}", save_rrd=False)
    #     rr.log(
    #         "room",
    #         rr.Boxes3D(
    #             centers=[(room_min + room_max) / 2],
    #             sizes=[room_max - room_min],
    #             colors=[(0.5, 0.5, 0.5, 0.5)],
    #         ),
    #     )
    
    # print(f'all ingredients: {room_config["ingredient"]}')

    fixtures = room_config["fixture"]
    for fixture_name, fixture_configs in fixtures.items():
        for idx, fixture_config in enumerate(fixture_configs):
            obj_path = os.path.join(config.SELECTED_ASSETS_PATH, "fixtures", f"{fixture_name}.obj")
            object_mesh = trimesh.load(obj_path)
            half_size = object_mesh.bounding_box.extents / 2 # extents 是边长不是半边长!!!
            fixture_vertices = object_mesh.vertices
            fixture_faces = object_mesh.faces

            fixture_translation = fixture_config["translation"]
            orientation = fixture_config["orientation"]

            center = object_mesh.bounding_box.centroid + fixture_translation # 位移之后的中心点
            if orientation == 90 or orientation == -90:
                half_size = half_size[[2, 1, 0]]  # 旋转之后的半边长

            offset = np.zeros(3)  # 初始化偏移数组为零
            for i in range(3):
                if center[i] - half_size[i] < room_min[i]:
                    offset[i] = center[i] - half_size[i] - room_min[i]
                elif center[i] + half_size[i] > room_max[i]:
                    offset[i] = center[i] + half_size[i] - room_max[i]

            rotation = R.from_euler("xyz", angles=[0, orientation, 0], degrees=True)
            fixture_vertices = (
                rotation.apply(fixture_vertices) + fixture_translation - offset
            )

            # update json, 把外面的物体往屋子里面移动
            room_config["fixture"][fixture_name][idx]["translation"] = (
                fixture_translation - offset
            )

            # # write rerun
            # if viz:
            #     rr_path = f"fixtures/{fixture_name}/{idx}/"
            #     rr.log(
            #         rr_path + "mesh",
            #         rr.Mesh3D(
            #             vertex_positions=fixture_vertices,
            #             triangle_indices=fixture_faces,
            #             vertex_normals=compute_vertex_normals(
            #                 fixture_vertices, fixture_faces
            #             ),
            #         ),
            #     )

            #     rr.log(
            #         rr_path + "bbox",
            #         rr.Transform3D(
            #             # scale=scale,
            #             translation=center - offset,
            #             rotation=rr.RotationAxisAngle(axis=(0, 1, 0), degrees=orientation),
            #         ),
            #     )
            #     rr.log(
            #         rr_path + "bbox",
            #         rr.Boxes3D(
            #             centers=object_mesh.bounding_box.centroid,
            #             sizes=object_mesh.bounding_box.extents,
            #         ),
            #     )

    # bbox loss
    val_bbox_loss=bbox_loss(room_config, viz)
    print(f'bbox_loss: {val_bbox_loss}')
    # print(f'cache_bbox: {cache_bbox}')

    # fatigue loss
    val_fatigue_loss=fatigue_loss(room_config, viz)
    print(f'fatigue_loss: {val_fatigue_loss}')

    # distance loss
    val_distance_loss=distance_loss(room_config, viz)
    print(f'distance_loss: {val_distance_loss}')
    
    # Add weights
    shelf_1=room_config['fixture']['shelf'][0]['translation'][[0,2]]
    shelf_2=room_config['fixture']['shelf'][1]['translation'][[0,2]]
    fixed_point_1=np.array([4,2])
    fixed_point_2=np.array([4,4])
    loss_add=np.linalg.norm(shelf_1-fixed_point_1)+np.linalg.norm(shelf_2-fixed_point_2)
    print(f'loss_add: {loss_add}')
    
    
    loss=weight_bbox*val_bbox_loss+weight_fatigue*val_fatigue_loss+weight_distance*val_distance_loss+weight_add*loss_add
    print(f'loss: {loss}, loss_bbox: {val_bbox_loss}, loss_fatigue: {val_fatigue_loss}, loss_distance: {val_distance_loss}, loss_add: {loss_add}\n')

    # if val_distance_loss<18:
    #     set_up_rerun('best')
    #     write_init_scene_human(room_config=room_config)
    #     exit(0)
        
    return loss


def wrap_loss_func(x: np.ndarray):
    scene=[x[0],x[1],x[2],x[3],x[4],x[5],
                    0,0,1,0,1,6,
                    0,0,0,1,1,0]
    scene=generate_one_room_config(scene)
    return loss_func(scene, viz=False)


def generate_one_room_config(params):
    '''
    params:
    [2,2]: shelf position
    [2]: wall shelf position
    [6]: which candidate: 0,1,2,3,4,5,6 for non-steak, 0,1 for steak
    [6]: which shelf: 0,1 for non-steak, 0 for steak, because only 2 shelf, 1 fridge
    '''
    tmp_room_config=copy.deepcopy(room_config_all)
    tmp_room_config['fixture']['shelf'][0]['translation'][0]=params[0]
    tmp_room_config['fixture']['shelf'][0]['translation'][2]=params[1]
    tmp_room_config['fixture']['shelf'][1]['translation'][0]=params[2]
    tmp_room_config['fixture']['shelf'][1]['translation'][2]=params[3]
    tmp_room_config['fixture']['wall_mounted_shelf'][0]['translation'][0]=params[4]
    tmp_room_config['fixture']['wall_mounted_shelf'][1]['translation'][0]=params[5]
    
    ingredients=['steak','onion','tomato','broccoli','bread','bacon']
    for index, ingredient in enumerate(ingredients):
        tmp_room_config['ingredient'][ingredient]=[room_config_all['ingredient'][ingredient][params[6+index]]]
        tmp_room_config['ingredient'][ingredient][0]['fixture_idx']=params[12+index]
        tmp_room_config['ingredient'][ingredient][0]['idx_for_viz']=params[6+index]
        # print(f'{ingredient} idx_for_viz: {params[6+index]}')
    
    return tmp_room_config

  
def main():
        
    room_config_all['fixture']['table'][0]['translation'][0]=3
    
    bad_scene=[1,1,1,4,2,2, # 一个很差的初始化场景
               1,0,1,1,4,5,
               0,0,1,0,1,1]
    bad_scene=generate_one_room_config(bad_scene)
    set_up_rerun('bad')
    loss_func(bad_scene, viz=True)
    write_init_scene_human(bad_scene)
    
    room_config_all['fixture']['fridge_base'][0]['translation'][2]=0
    room_config_all['fixture']['worktop'][0]['translation'][2]=8
    room_config_all['fixture']['stove_top'][0]['translation'][2]=0.7+1.3
    room_config_all['fixture']['hood'][0]['translation'][2]=0.7+1.3
    room_config_all['fixture']['table'][0]['translation'][0]=2.5

    good_scene=[4,2,4,4,3,2, # 好的场景
                0,0,1,0,1,6,
                0,0,0,1,1,0]
    good_scene=generate_one_room_config(good_scene)
    set_up_rerun('good')
    loss_func(good_scene, viz=True)
    write_init_scene_human(good_scene)
    
    exit(0)
    
    
    
    x0 = [4,2,4,4,2,2]
    sigma0 = 1

    fun = wrap_loss_func  # we could use `functools.partial(cma.ff.elli, cond=1e4)` to change the condition number to 1e4
    def constraints(x):
        return [x[0] - 6, x[1] - 6, x[2] - 6, x[3] - 6, x[4] - 6, x[5] - 6,
                -x[0], -x[1], -x[2], -x[3], -x[4], -x[5]]  # constrain the second variable to <= -1, the second constraint is superfluous
    cfun = cma.ConstrainedFitnessAL(fun, constraints)  # unconstrained function with adaptive Lagrange multipliers

    x, es = cma.fmin2(cfun, x0, sigma0, {'tolstagnation': 0}, callback=cfun.update)
    x = es.result.xfavorite  # the original x-value may be meaningless
    constraints(x)  # show constraint violation values

    best_scene=generate_one_room_config(x)
    set_up_rerun(app_id='best')
    loss_func(best_scene, viz=False)
    write_init_scene_human(best_scene)


if __name__ == "__main__":
    main()
