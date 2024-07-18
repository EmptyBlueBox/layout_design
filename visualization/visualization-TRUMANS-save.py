import smplx
import argparse
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from rerun.datatypes import Quaternion
import rerun as rr
import trimesh
import numpy
import pickle
from utils.mesh_utils import compute_vertex_normals


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


TRUMANS_PATH = '/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS'
device = setup_device()

VISUALIZATION = True  # 是否可视化
SAVE = True  # 是否保存 human 和 object 的数据, 如果保存, 那么就是从末端开始往前选一定长度的帧, 否则从整体均匀选一定长度的帧
OBJECT_ONLY = True  # 是否只选择物体

SEG_NUM = 1  # 可视化哪个seg
SET_FRAME_NUM = 50  # 一个seg中试图选择的帧数
START_RATIO = 0.0  # 选择开始的帧数比例, 如果是0.44, 那么就是从44%的位置开始, 在 SAVE=True 的情况下, 是没用的
END_RATIO = 0.455  # 选择停止的帧数比例, 如果是0.48, 那么就是到48%的位置结束, 在 SAVE=True 的情况下, 从这个地方往前选 SET_FRAME_NUM 长度的帧
# start_frame_ratio = 0.  # 选择开始的帧数比例
# end_frame_ratio = 1.  # 选择停止的帧数比例

SAVE_PATH = f'/Users/emptyblue/Documents/Research/layout_design/dataset/chair-vanilla/seat_{SEG_NUM}-frame_num_{SET_FRAME_NUM}'
SAVE_PATH = f'/Users/emptyblue/Documents/Research/layout_design/dataset/data_pair-hand_picked/seat_{SEG_NUM}-frame_num_{SET_FRAME_NUM}'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


seg_begin_list = np.load(TRUMANS_PATH+'/seg_begin.npy')
seg_begin = seg_begin_list[SEG_NUM]  # 当前seg的开始帧
seg_end = seg_begin_list[SEG_NUM+1]-1 if SEG_NUM+1 < seg_begin_list.shape[0] else len(seg_begin_list)-1  # 当前seg的结束帧
seg_frame_num = seg_end - seg_begin + 1

select_start = int(START_RATIO*seg_end+(1-START_RATIO)*seg_begin)
select_end = int(END_RATIO*seg_end+(1-END_RATIO)*seg_begin)

if SAVE:  # 需要保存的时候从末端开始选一定长度的帧
    max_frame_num = min(SET_FRAME_NUM, int(seg_frame_num*END_RATIO))  # 实际的帧数
    print(f'max_frame_num: {max_frame_num}')
    selected_frame = np.arange(select_end-max_frame_num+1, select_end+1)
    frames_per_second = 60
    print(f'frames_per_second: {frames_per_second}')
else:  # 不需要保存的时候从整体均匀选一定长度的帧
    max_frame_num = min(SET_FRAME_NUM, int(seg_frame_num*(END_RATIO-START_RATIO)))  # 实际的帧数
    print(f'max_frame_num: {max_frame_num}')
    selected_frame = np.linspace(select_start,
                                 select_end,
                                 max_frame_num,
                                 dtype=int)
    frames_per_second = max_frame_num / (seg_frame_num * (END_RATIO - START_RATIO)) * 60
    print(f'frames_per_second: {frames_per_second}')


def load_human():

    human_model = smplx.create(model_path='/Users/emptyblue/Documents/Research/HUMAN_MODELS',
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=max_frame_num)

    human_params = {
        'poses': np.load(TRUMANS_PATH+'/human_pose.npy')[selected_frame].reshape(-1, 21, 3),  # (max_frame_num, 21, 3)
        'orientation': np.load(TRUMANS_PATH+'/human_orient.npy')[selected_frame].reshape(-1, 3),  # (max_frame_num, 3)
        'translation': np.load(TRUMANS_PATH+'/human_transl.npy')[selected_frame].reshape(-1, 3),  # (max_frame_num, 3)
    }

    # print(f'smpl_params.pose: {smpl_params["poses"].shape}')
    # print(f'smpl_params.orientation: {smpl_params["orientation"].shape}')
    # print(f'smpl_params.translation: {smpl_params["translation"].shape}')

    output = human_model(body_pose=torch.tensor(human_params['poses'], dtype=torch.float32),
                         global_orient=torch.tensor(human_params['orientation'], dtype=torch.float32),
                         transl=torch.tensor(human_params['translation'], dtype=torch.float32))
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    # 对每一帧计算顶点法向量
    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)

    # print(f'vertices.shape: {vertices.shape}')
    # print(f'body_model.faces.shape: {human_model.faces.shape}')
    # print(f'vertex_normals.shape: {vertex_normals.shape}')

    human_params['vertices'] = vertices
    human_params['faces'] = faces
    human_params['vertex_normals'] = vertex_normals

    return human_params


def load_object():
    seg_name_list = np.load(TRUMANS_PATH+'/seg_name.npy')
    seg_name = seg_name_list[seg_begin]
    object: dict = numpy.load(TRUMANS_PATH+f'/Object_all/Object_pose/{seg_name}.npy', allow_pickle=True).item()  # 这是一个被{}包起来的字典, 要做一下解包 .item()
    # 只要名字里有 chair 的
    if OBJECT_ONLY:
        object = {key: object[key] for key in object.keys() if 'chair' in key}

    for key in object.keys():
        object[key]['rotation'] = np.array(object[key]['rotation'])[selected_frame-seg_begin]
        object[key]['location'] = np.array(object[key]['location'])[selected_frame-seg_begin]
        object_name = key
        decimated_mesh_path = f'../dataset/TRUMANS/Object_all/Object_mesh_decimated/{object_name}.obj'
        # 如果已经有了decimated mesh, 就直接读取, 否则读取原始 mesh
        if os.path.exists(decimated_mesh_path):
            decimated_mesh = trimesh.load(decimated_mesh_path)
            object[key]['vertices'] = decimated_mesh.vertices
            object[key]['faces'] = decimated_mesh.faces
            object[key]['vertex_normals'] = compute_vertex_normals(object[key]['vertices'], object[key]['faces'])
        else:
            object_obj_path = f'../dataset/TRUMANS/Object_all/Object_mesh/{object_name}.obj'
            original_mesh = trimesh.load(object_obj_path)
            object[key]['vertices'] = original_mesh.vertices
            object[key]['faces'] = original_mesh.faces
            object[key]['vertex_normals'] = compute_vertex_normals(object[key]['vertices'], object[key]['faces'])

    return object


def load_scene():
    if SAVE:
        return None
    scene_flag = np.load(TRUMANS_PATH+'/scene_flag.npy')[seg_begin]  # 当前seg的场景标志
    scene_list = np.load(TRUMANS_PATH+'/scene_list.npy')  # 一个包含所有场景的列表
    scene_name = scene_list[scene_flag]  # 一个场景的名字
    print(f'scene_name: {scene_name}\n')
    scene_path = f'../dataset/TRUMANS/Scene_mesh_decimated/{scene_name}.obj'
    scene_mesh = trimesh.load(scene_path)  # 一个场景的 mesh

    scene = {
        'vertices': scene_mesh.vertices,
        'faces': scene_mesh.faces,
        'vertex_normals': compute_vertex_normals(scene_mesh.vertices, scene_mesh.faces)
    }

    return scene


def write_rerun(human: dict, object: dict, scene: dict):
    print('writing rerun...')
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, f'TRUMANS seg: {SEG_NUM}, frame_num: {max_frame_num}')
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)

    if scene is not None:
        rr.log(
            'scene',
            rr.Mesh3D(
                vertex_positions=scene['vertices'],
                triangle_indices=scene['faces'],
                vertex_normals=scene['vertex_normals'],
            ),
        )
        print(f'scene vertices: {scene["vertices"].shape[0]}, faces: {scene["faces"].shape[0]}')
    print(f'human vertices: {human["vertices"][0].shape[0]}, faces: {human["faces"].shape[0]}')
    for key in object.keys():
        print(f'{key} vertices: {object[key]["vertices"].shape[0]}, faces: {object[key]["faces"].shape[0]}')

    for i in range(max_frame_num):
        time = i / frames_per_second
        rr.set_time_seconds("stable_time", time)

        rr.log(
            'human',
            rr.Mesh3D(
                vertex_positions=human['vertices'][i],
                triangle_indices=human['faces'],
                vertex_normals=human['vertex_normals'][i],
            ),
        )

        for key in object.keys():  # 一个frame中遍历所有object: 人物, 椅子, 桌子等
            rr_transform = rr.Transform3D(
                rotation=Quaternion(xyzw=R.from_euler('xyz', object[key]['rotation'][i]).as_quat()),
                translation=object[key]['location'][i],
            )

            rr.log(f'object/{key}', rr_transform)
            rr.log(f'object/{key}',
                   rr.Mesh3D(
                       vertex_positions=object[key]['vertices'],
                       triangle_indices=object[key]['faces'],
                       vertex_normals=object[key]['vertex_normals'],
                   ),
                   )

    rr.script_teardown(args)
    print('write rerun done!\n')


def save_data(human: dict, object: dict, scene: dict):
    print('saving human and object mesh...')
    print(f'human vertices: {human["vertices"][0].shape[0]}, faces: {human["faces"].shape[0]}')
    for key in object.keys():
        print(f'{key} vertices: {object[key]["vertices"].shape[0]}, faces: {object[key]["faces"].shape[0]}')

    human_params = {'poses': human['poses'], 'orientation': human['orientation'], 'translation': human['translation']}
    pickle.dump(human_params, open(f'{SAVE_PATH}/human-params.pkl', 'wb'))  # 保存 smpl_params
    print(f'human params saved to {SAVE_PATH}/human-params.pkl, \nkeys: {human_params.keys()}')

    # human_mesh = {'vertices': human['vertices'], 'faces': human['faces'], 'vertex_normals': human['vertex_normals']}
    # pickle.dump(human, open(f'{SAVE_PATH}/human-mesh.pkl', 'wb'))  # 保存 human
    # print(f'human mesh saved to {SAVE_PATH}/human-mesh.pkl, \nkeys: {human.keys()}\n')

    object_params = {key: {'rotation': object[key]['rotation'], 'location': object[key]['location']} for key in object.keys()}
    pickle.dump(object_params, open(f'{SAVE_PATH}/object-params.pkl', 'wb'))  # 保存 object_params
    print(
        f'object params saved to {SAVE_PATH}/object-params.pkl, \nobjects: {object_params.keys()}, keys: {object_params[list(object_params.keys())[0]].keys()}\n')

    # object_mesh = {key: {'vertices': object[key]['vertices'], 'faces': object[key]['faces'],
    #                      'vertex_normals': object[key]['vertex_normals']} for key in object.keys()}
    # pickle.dump(object_mesh, open(f'{SAVE_PATH}/object-mesh.pkl', 'wb'))  # 保存 object
    # print(
    #     f'object mesh saved to {SAVE_PATH}/object-mesh.pkl, \nobjects: {object.keys()}, keys: {object[list(object.keys())[0]].keys()}\n')

    # print(f'saved mesh frame_num: {max_frame_num}, frames_per_second: {frames_per_second}')
    print('save human and object mesh done!\n')


def main():
    human = load_human()
    object = load_object()
    scene = load_scene()
    scene = None  # 不可视化 scene
    if VISUALIZATION:
        write_rerun(human=human, object=object, scene=scene)
    if SAVE:
        save_data(human=human, object=object, scene=scene)


if __name__ == '__main__':
    main()
