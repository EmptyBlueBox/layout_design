import argparse
import numpy as np
import cma
import os
import regex as re
import pickle
import config
from scipy.spatial.transform import Rotation as R
from utils.mesh_utils import compute_vertex_normals, query_sdf_normalized, query_bounding_box_normalized, farthest_point_sampling
import trimesh
import smplx
import torch
from tqdm import tqdm
import rerun as rr
import time

visualize = True


def load_data(step=30):
    DATA_FOLDER = '/Users/emptyblue/Documents/Research/layout_design/dataset/data_pair-hand_picked'
    subdata_list = os.listdir(DATA_FOLDER)
    subdata_list.sort()
    subdata_list = [name for name in subdata_list if re.match(r'seg_\d+-obj_\w+-end_\d+\.\d+-len_\d+', name)]  # seg_3-obj_fridge-end_0.382-len_130

    data = {}
    for subdata_name in subdata_list:
        subdata_object_path = os.path.join(DATA_FOLDER, subdata_name, 'object-params.pkl')
        object_params = pickle.load(open(subdata_object_path, 'rb'))
        data[tuple(object_params.keys())] = []

    for subdata_name in subdata_list:
        subdata_human_path = os.path.join(DATA_FOLDER, subdata_name, 'human-params.pkl')
        subdata_object_path = os.path.join(DATA_FOLDER, subdata_name, 'object-params.pkl')
        human_params = pickle.load(open(subdata_human_path, 'rb'))
        object_params = pickle.load(open(subdata_object_path, 'rb'))
        data_entry = {'human_params': human_params, 'object_params': object_params}
        # data[tuple(object_params.keys())].append(data_entry)
        data[tuple(object_params.keys())] = [data_entry]  # 只取第一个数据

    for key in data.keys():
        for i in range(len(data[key])):
            for single_obj_name in key:
                data[key][i]['object_params'][single_obj_name]['location'] = data[key][i]['object_params'][single_obj_name]['location'][::step]
                data[key][i]['object_params'][single_obj_name]['rotation'] = data[key][i]['object_params'][single_obj_name]['rotation'][::step]
            data[key][i]['human_params']['translation'] = data[key][i]['human_params']['translation'][::step]
            data[key][i]['human_params']['orientation'] = data[key][i]['human_params']['orientation'][::step]
            data[key][i]['human_params']['poses'] = data[key][i]['human_params']['poses'][::step]
            # 人体和物体的 x z 位置归零
            base_obj_name = key[0]
            for single_obj_name in key:
                if single_obj_name != base_obj_name:
                    data[key][i]['object_params'][single_obj_name]['location'][:, [0, 2]
                                                                               ] -= data[key][i]['object_params'][base_obj_name]['location'][0, [0, 2]]
            data[key][i]['human_params']['translation'][:, [0, 2]] -= data[key][i]['object_params'][base_obj_name]['location'][0, [0, 2]]
            data[key][i]['object_params'][base_obj_name]['location'][:, [0, 2]] -= data[key][i]['object_params'][base_obj_name]['location'][0, [0, 2]]
    return data


def print_data(data: dict):
    print(f'All objects: {data.keys()}')
    for key in data.keys():
        print(f'Object {key} has {len(data[key])} data entries')
    print(f"One data entry's human_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['human_params'].keys()}")
    print(f"One data entry's object_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['object_params'].keys()}")
    # Print trajectory scale
    for key in data.keys():
        trajectory = data[key][0]['human_params']['translation']
        delta_x = trajectory[0][0]-trajectory[-1][0]
        delta_y = trajectory[0][1]-trajectory[-1][1]
        delta_z = trajectory[0][2]-trajectory[-1][2]
        scale = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        # print(f'Object {key}, trajectory scale: {scale}, delta_x: {delta_x}, delta_y: {delta_y}, delta_z: {delta_z}')
    print(f"One data entry\'s object_params has key: {data[('cabinet_base_01', 'cabinet_door_01')][0]['object_params']['cabinet_base_01'].keys()}\n")


class sample_query_point:
    def __init__(self, data, visualize=False):
        self.data = data
        self.visualize = visualize

        self.object_num = len(data)
        self.obj_name = list(data.keys())
        self.cache_zero_query_point = None
        self.max_frame = None

        self.fps = 30

    def sample_point_from_motion(self,
                                 object_params,
                                 human_params,
                                 delta_x,
                                 delta_theta,
                                 object_mesh_path=config.OBJECT_DECIMATED_PATH,
                                 max_frame_num=500,
                                 human_query_max=500,
                                 object_query_max=300,
                                 over_sample_factor=3,
                                 save=False):
        R_delta = R.from_rotvec(np.array([0, delta_theta, 0]))
        frame_num = human_params['poses'].shape[0]
        # 计算物体旋转之后的参数和顶点, 基础物体自己旋转即可
        object_base = next(iter(object_params.keys()))
        object_mesh = {}
        for key in object_params.keys():
            object_orientation = R_delta*R.from_euler('xyz', object_params[key]['rotation'])
            if key == object_base:
                object_translation = object_params[key]['location'] + delta_x
            else:
                object_translation = R_delta.apply(object_params[key]['location']-object_params[object_base]
                                                   ['location'])+object_params[object_base]['location']+delta_x
            decimated_mesh_path = os.path.join(object_mesh_path, key + '.obj')
            decimated_mesh = trimesh.load(decimated_mesh_path)
            object_mesh[key] = {}  # 初始化一个子物体的字典
            object_mesh[key]['vertices'] = np.zeros((max_frame_num, decimated_mesh.vertices.shape[0], 3))
            for t in range(frame_num):
                object_mesh[key]['vertices'][t] = object_orientation[t].apply(decimated_mesh.vertices) + object_translation[t]
            for t in range(frame_num, max_frame_num):
                object_mesh[key]['vertices'][t] = object_mesh[key]['vertices'][frame_num-1]
            object_mesh[key]['faces'] = decimated_mesh.faces
            object_mesh[key]['vertex_normals'] = np.zeros_like(object_mesh[key]['vertices'])
            for t in range(max_frame_num):
                object_mesh[key]['vertex_normals'][t] = compute_vertex_normals(object_mesh[key]['vertices'][t], object_mesh[key]['faces'])

        object_base = next(iter(object_params.keys()))
        # 计算人体旋转之后的参数
        human_orientation = R_delta*R.from_rotvec(human_params['orientation'])
        human_orientation = human_orientation.as_rotvec()
        human_translation = R_delta.apply(
            human_params['translation']-object_params[object_base]['location'])+object_params[object_base]['location']+delta_x
        # print(
        #     f"test human: {human_params['translation'][:5]}, delta_x: {delta_x}, object_params[object_base]['location']: {object_params[object_base]['location'][:5]}")
        # 计算人体的顶点
        human_model = smplx.create(model_path=config.SMPL_MODEL_PATH,
                                   model_type='smplx',
                                   gender='neutral',
                                   use_face_contour=False,
                                   num_betas=10,
                                   num_expression_coeffs=10,
                                   ext='npz',
                                   batch_size=frame_num)
        output = human_model(body_pose=torch.tensor(human_params['poses'], dtype=torch.float32),
                             global_orient=torch.tensor(human_orientation, dtype=torch.float32),
                             transl=torch.tensor(human_translation, dtype=torch.float32))
        vertices_0 = output.vertices.detach().cpu().numpy()
        vertices = np.zeros((max_frame_num, vertices_0.shape[1], 3))
        vertices[:frame_num] = vertices_0
        vertices[frame_num:] = vertices[frame_num-1]
        faces = human_model.faces
        vertex_normals = np.zeros_like(vertices)
        for i in range(vertices.shape[0]):
            vertex_normals[i] = compute_vertex_normals(vertices[i], faces)
        human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

        if self.visualize and save:
            for i in range(max_frame_num):
                time = i / self.fps
                rr.set_time_seconds("stable_time", time)

                # 画人
                rr.log(
                    f'{object_base}/human',
                    rr.Mesh3D(
                        vertex_positions=human_mesh['vertices'][i],
                        triangle_indices=human_mesh['faces'],
                        vertex_normals=human_mesh['vertex_normals'][i],
                    ),
                )

                # 画物体
                for key in object_mesh.keys():  # 一个frame中遍历所有object: 人物, 椅子, 桌子等
                    rr.log(f'{object_base}/object/{key}',
                           rr.Mesh3D(
                               vertex_positions=object_mesh[key]['vertices'][i],
                               triangle_indices=object_mesh[key]['faces'],
                               vertex_normals=object_mesh[key]['vertex_normals'][i],
                           ),
                           )

        object_query_points = np.concatenate([object_mesh[key]['vertices'] for key in object_mesh.keys()], axis=1)
        human_query_points = human_mesh['vertices']
        if human_query_points.shape[1] > human_query_max:
            # 随机选择3*human_query_max个点
            over_sample_num = min(over_sample_factor*human_query_max, human_query_points.shape[1])
            human_query_points = human_query_points[:, np.random.choice(human_query_points.shape[1], over_sample_num, replace=False)]
            human_query_points = farthest_point_sampling(human_query_points, human_query_max)
        if object_query_points.shape[1] > object_query_max:
            # 随机选择3*object_query_max个点
            over_sample_num = min(over_sample_factor*object_query_max, object_query_points.shape[1])
            object_query_points = object_query_points[:, np.random.choice(object_query_points.shape[1], over_sample_num, replace=False)]
            object_query_points = farthest_point_sampling(object_query_points, object_query_max)
        human_query_points = human_query_points.reshape(-1, 3)
        object_query_points = object_query_points.reshape(-1, 3)

        all_query = np.concatenate([human_query_points, object_query_points], axis=0)
        # print('all_query:', all_query.shape)

        if self.visualize and save:
            rr.log(f'{object_base}/query_points', rr.Points3D(all_query))

        return all_query

    def rotate_cache(self, x):
        query_point = {}
        for i in range(self.object_num):
            key = self.obj_name[i]
            query_point[key] = []
            delta_x = np.array([x[i][0], 0, x[i][1]])
            R_delta = R.from_rotvec(np.array([0, x[i][2], 0]))
            for i in range(len(self.cache_zero_query_point[key])):
                query = self.cache_zero_query_point[key][i]
                query_point[key].append(R_delta.apply(query) + delta_x)

        if self.visualize:
            for i in range(self.object_num):
                # print(f'object: {self.obj_name[i]}, x: {x[i]}, query_point: {query_point[self.obj_name[i]][0][:10]}')
                rr.log(f'{self.obj_name[i][0]}/query_points', rr.Points3D(query_point[self.obj_name[i]][0]))

        return query_point

    def __call__(self,
                 x,
                 save=False  # 是否保存 rerun 文件, 只有在优化到 loss 为 0 时才需要保存
                 ):
        '''
        x: (K, 3) 二维数组, K 为物体数量, 3 为每个物体的xz位移和y旋转
        '''
        max_frame = 0
        for key in self.data.keys():
            for i in range(len(self.data[key])):
                length = self.data[key][i]['human_params']['translation'].shape[0]
                length = self.data[key][i]['human_params']['orientation'].shape[0]
                length = self.data[key][i]['human_params']['poses'].shape[0]
                for single_obj_name in key:
                    length = self.data[key][i]['object_params'][single_obj_name]['location'].shape[0]
                    length = self.data[key][i]['object_params'][single_obj_name]['rotation'].shape[0]
                max_frame = max(max_frame, length)
        self.max_frame = max_frame
        # print(f'max_frame: {max_frame}')

        if self.visualize:
            print('writing rerun...')
            parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
            rr.script_add_args(parser)
            args = parser.parse_args()
            rr.script_setup(args, f'CMAES-{config.ROOM_SHAPE_X}x{config.ROOM_SHAPE_Z}-{save}')
            rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
            rr.set_time_seconds("stable_time", 0)

            # 画墙
            rr.log(
                "wall",
                rr.Boxes3D(
                    centers=[config.ROOM_SHAPE_X/2, .8, config.ROOM_SHAPE_Z/2],
                    half_sizes=[config.ROOM_SHAPE_X/2, .8, config.ROOM_SHAPE_Z/2],
                    radii=0.01,
                    colors=(0, 0, 255),
                    labels="blue",
                ),
            )

        test = time.time()
        if self.cache_zero_query_point is None:
            print('now caching query point')
            query_point = {}
            for i in range(self.object_num):
                query_point[self.obj_name[i]] = []
                for j in range(len(self.data[self.obj_name[i]])):
                    # print(f'object: {self.obj_name[i]}, x: {x[i]}')
                    human_params = self.data[self.obj_name[i]][j]['human_params']
                    object_params = self.data[self.obj_name[i]][j]['object_params']
                    query = self.sample_point_from_motion(object_params,
                                                          human_params,
                                                          np.array([0, 0, 0]),
                                                          0,
                                                          max_frame_num=max_frame,
                                                          save=save)
                    query_point[self.obj_name[i]].append(query)
            self.cache_zero_query_point = query_point

        if save:
            query_point = {}
            for i in range(self.object_num):
                query_point[self.obj_name[i]] = []
                for j in range(len(self.data[self.obj_name[i]])):
                    # print(f'object: {self.obj_name[i]}, x: {x[i]}')
                    human_params = self.data[self.obj_name[i]][j]['human_params']
                    object_params = self.data[self.obj_name[i]][j]['object_params']
                    query = self.sample_point_from_motion(object_params,
                                                          human_params,
                                                          np.array([x[i][0], 0, x[i][1]]),
                                                          x[i][2],
                                                          max_frame_num=max_frame,
                                                          save=save)
                    query_point[self.obj_name[i]].append(query)
            self.cache_zero_query_point = query_point
        else:
            query_point = self.rotate_cache(x)
        # print(f'sample query point time: {time.time()-test}, using cache: {self.cache_zero_query_point is not None}')

        if self.visualize:
            rr.script_teardown(args)
            print('write rerun done!\n')

        return query_point


class objective_function:
    def __init__(self,
                 data,
                 sampler,
                 step=10,
                 obj_loss_scale=10,
                 wall_loss_scale=1,
                 dis_loss_scale=5,
                 object_point=1000,
                 human_point=2000,
                 oversample_factor=2):
        self.data = data
        self.sampler = sampler
        self.step = step
        self.obj_loss_scale = obj_loss_scale
        self.wall_loss_scale = wall_loss_scale
        self.dis_loss_scale = dis_loss_scale
        self.object_point = object_point
        self.human_point = human_point
        self.oversample_factor = oversample_factor

        self.object_num = len(data)
        self.obj_name = list(data.keys())  # [(base, move), ()]
        self.query_point = None

    # 计算物体人点云和另一个物体之间的碰撞
    def loss_motion_obj(self, obj_name1, obj_name2, info1, info2):
        obj_num_1 = len(obj_name1)  # 一个物体组中的物体数量
        obj_num_2 = len(obj_name2)
        motion_num1 = len(self.data[obj_name1])  # 一个物体组中的运动轨迹数量
        motion_num2 = len(self.data[obj_name2])

        translation_1 = np.array([info1[0], 0, info1[1]])
        orientation_1 = R.from_rotvec(np.array([0, info1[2], 0]))
        translation_2 = np.array([info2[0], 0, info2[1]])
        orientation_2 = R.from_rotvec(np.array([0, info2[2], 0]))

        query_point_1 = self.query_point[obj_name1]
        query_point_2 = self.query_point[obj_name2]

        # 计算两个物体之间的碰撞
        motion_clip_penetration = []  # 每个元素是一个 motion clip 和另一个物体的碰撞深度总和, 最后只需要取最大值即可

        for i in range(motion_num1):  # 对第一个物体组的每一个 motion trajectory 是一个点云
            for j in range(motion_num2):  # 对第二个物体组的每一个 motion trajectory 是一个点云
                point_cloud_1_when_2_normal = orientation_2.apply(query_point_1[i]-translation_2)
                # signed_distence = query_sdf_normalized(point_cloud_1_when_2_normal, obj_name2[0])  # 一个点云和一个物体的碰撞深度
                signed_distence = query_bounding_box_normalized(point_cloud_1_when_2_normal, obj_name2[0])  # 一个点云和一个物体的碰撞深度
                negative_mask = signed_distence < 0
                loss_1 = -np.sum(signed_distence[negative_mask])
                # if loss_1 > 1e-6:
                #     rr.log(f'TEST/{obj_name1[0]}-{obj_name2[0]}-2_normal', rr.Points3D(point_cloud_1_when_2_normal[negative_mask]))
                #     rr.log(f'TEST/{obj_name1[0]}-{obj_name2[0]}', rr.Points3D(query_point_1[i][negative_mask]))

                point_cloud_2_when_1_normal = orientation_1.apply(query_point_2[j]-translation_1)
                # signed_distence = query_sdf_normalized(point_cloud_2_when_1_normal, obj_name1[0])
                signed_distence = query_bounding_box_normalized(point_cloud_2_when_1_normal, obj_name1[0])
                negative_mask = signed_distence < 0
                loss_2 = -np.sum(signed_distence[negative_mask])
                # if loss_2 > 1e-6:
                #     rr.log(f'TEST/{obj_name2[0]}-{obj_name1[0]}-2_normal', rr.Points3D(point_cloud_2_when_1_normal[negative_mask]))
                #     rr.log(f'TEST/{obj_name2[0]}-{obj_name1[0]}', rr.Points3D(query_point_2[j][negative_mask]))

                motion_clip_penetration.append(loss_1+loss_2)

        motion_clip_penetration = np.array(motion_clip_penetration)
        loss = np.mean(motion_clip_penetration)
        # print(f'obj1: {obj_name1}, obj2: {obj_name2}, loss: {loss}')
        return loss

    def loss_motion_wall(self, motion_name, info):
        query_point = self.query_point[motion_name]
        motion_num = len(self.data[motion_name])  # 一个物体组中的运动轨迹数量
        translation_1 = np.array([info[0], 0, info[1]])

        # 计算人体和墙之间的碰撞
        x_min = 0
        x_max = config.ROOM_SHAPE_X
        z_min = 0
        z_max = config.ROOM_SHAPE_Z

        motion_clip_penetration = []  # 每个元素是一个 motion clip 和墙的碰撞深度总和, 最后只需要取最大值即可
        for i in range(motion_num):  # 对第一个物体组的每一个 motion trajectory 是一个点云
            point_cloud = query_point[i]
            signed_distence_x = np.maximum(point_cloud[:, 0]-x_max, x_min-point_cloud[:, 0])
            signed_distence_z = np.maximum(point_cloud[:, 2]-z_max, z_min-point_cloud[:, 2])
            signed_distence = np.maximum(signed_distence_x, signed_distence_z)
            penetration_mask = signed_distence > 0

            weight = np.exp(-np.linalg.norm(point_cloud[penetration_mask]-translation_1, axis=1))
            loss = np.sum(signed_distence[penetration_mask]*weight)
            motion_clip_penetration.append(loss)

        motion_clip_penetration = np.array(motion_clip_penetration)
        loss = np.mean(motion_clip_penetration)
        # print(f'motion_name: {motion_name}, loss: {loss}')
        return loss

    def loss_distance(self, x):
        end = np.array([[self.data[key][0]['human_params']['translation'][-1][0], 0, self.data[key]
                         [0]['human_params']['translation'][-1][2]] for key in self.obj_name])
        start = np.array([[self.data[key][0]['human_params']['translation'][0][0], 0, self.data[key]
                           [0]['human_params']['translation'][0][2]] for key in self.obj_name])
        R_delta = R.from_rotvec(np.array([np.zeros(self.object_num), x[:, 2], np.zeros(self.object_num)]).T)
        offest = np.array([x[:, 0], np.zeros(self.object_num), x[:, 1]]).T
        # print(
        #     f'start shape: {end.shape}, end shape: {start.shape}, rot shape: {np.array([np.zeros(self.object_num), x[:, 2], np.zeros(self.object_num)]).T}, offest shape: {offest.shape}')
        end = R_delta.apply(end)+offest
        start = R_delta.apply(start)+offest
        # print(f'start: {end}, end: {start}')

        loss = 0
        for i in range(self.object_num):
            j = (i+1) % self.object_num
            loss += np.linalg.norm(end[i]-start[j])
        return loss

    def __call__(self, x):
        print('Start objective_function')
        obj_start = time.time()

        x = x.reshape(self.object_num, 3)
        test = time.time()
        self.query_point = self.sampler(x)  # 采样点云, 同时进行了可视化
        # print(f'sample query point time: {time.time()-test}')

        # 计算 motion clip 和所有物体之间的碰撞
        test = time.time()
        loss_obj2obj = 0
        for i in range(self.object_num):
            for j in range(i+1, self.object_num):
                loss_obj2obj += self.loss_motion_obj(self.obj_name[i], self.obj_name[j], x[i], x[j])
        # print(f'loss_obj2obj: {loss_obj2obj}, time: {time.time()-test}')

        # 计算所有动作和墙之间的碰撞
        test = time.time()
        loss_human_obj = 0
        for i in range(self.object_num):
            loss_human_obj += self.loss_motion_wall(self.obj_name[i], x[i])
        # print(f'loss_human_obj: {loss_human_obj}, time: {time.time()-test}')

        # 计算所有物体之间的距离
        test = time.time()
        loss_distance = self.loss_distance(x)
        # print(f'loss_distance: {loss_distance}, time: {time.time()-test}')

        print(f'loss_motion_obj: {loss_obj2obj}, loss_motion_wall: {loss_human_obj}, loss_distance: {loss_distance}')

        loss = self.obj_loss_scale * loss_obj2obj + self.wall_loss_scale * loss_human_obj + self.dis_loss_scale * loss_distance
        print(f'End objective_function, time: {time.time()-obj_start}\n')

        if np.abs(loss_obj2obj * self.obj_loss_scale + self.wall_loss_scale * loss_human_obj) < 0.1 and loss_distance < 5:
            self.sampler(x, save=True)  # 重新可视化
            print('loss is 0, done.')
            exit(0)

        # exit(0)  # 仅测试
        return loss


def constraints(x):
    xmin = 0
    xmax = config.ROOM_SHAPE_X
    zmin = 0
    zmax = config.ROOM_SHAPE_Z
    output = []
    x = x.reshape(-1, 3)
    # print(f'x shape: {x.shape}')
    for i in range(x.shape[0]):
        output.append(- x[i][0] + xmin)  # x >= xmin
        output.append(x[i][0] - xmax)  # x <= xmax
        output.append(- x[i][1] + zmin)  # z >= zmin
        output.append(x[i][1] - zmax)   # z <= zmax
        output.append(- x[i][2] - np.pi)  # orientation >= -pi
        output.append(x[i][2] - np.pi)   # orientation <= pi

    return output


def optimize(data: dict):
    object_num = len(data)
    translation_init = np.random.rand(object_num, 2)*np.array([config.ROOM_SHAPE_X, config.ROOM_SHAPE_Z])
    orientation_init = np.random.rand(object_num, 1)*2*np.pi-np.pi
    # print(f'Initial translation: {translation_init}, orientation: {orientation_init}')
    x0 = np.concatenate((translation_init, orientation_init), axis=1).reshape(-1)
    sigma0 = 1

    sampler = sample_query_point(data, visualize=visualize)
    obj = objective_function(data,
                             sampler,
                             obj_loss_scale=10,
                             step=100,
                             object_point=3000)
    cfun = cma.ConstrainedFitnessAL(obj, constraints)  # unconstrained function with adaptive Lagrange multipliers

    x, es = cma.fmin2(cfun, x0, sigma0, {'tolstagnation': 0}, callback=cfun.update)
    x = es.result.xfavorite  # the original x-value may be meaningless
    constraints(x)  # show constraint violation values

    output = {
        'translation': x[:, :2],
        'orientation': x[:, 2]
    }
    return output


if __name__ == '__main__':
    data = load_data()
    print_data(data=data)
    output = optimize(data)
    print(output)
