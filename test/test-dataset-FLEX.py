import trimesh
import rerun as rr
import numpy as np
import os
from utils.mesh_utils import compute_vertex_normals
from scipy.spatial.transform import Rotation as R
import config
import smplx
import torch
import sys
sys.path.append('/Users/emptyblue/Documents/Research/FLEX/')


relationships_path = '/Users/emptyblue/Documents/Research/layout_design/test/test-FLEX/ReplicaGrasp/dset_info.npz'

holder_path = '/Users/emptyblue/Documents/Research/layout_design/test/test-FLEX/ReplicaGrasp/receptacles.npz'

obj_mesh_root = '/Users/emptyblue/Documents/Research/layout_design/test/test-FLEX/obj/contact_meshes'
obj_bps_path = '/Users/emptyblue/Documents/Research/layout_design/test/test-FLEX/obj/bps.npz'  # 不用修改, 对所有物体都是一样的
obj_info_path = '/Users/emptyblue/Documents/Research/layout_design/test/test-FLEX/obj/obj_info.npy'  # 没用


INDEX = 100

motion_path = '/Users/emptyblue/Documents/Research/layout_design/test/test-FLEX/all_0.npz'


def set_up_rerun(save_rrd=False):
    rr.init('Visualization: Kitchen', spawn=not save_rrd)
    if save_rrd:
        save_path = 'rerun-tmp'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        rr.save(os.path.join(save_path, 'Kitchen.rrd'))
    # rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)


def write_obj():
    relationships = np.load(relationships_path, allow_pickle=True)
    relationships_name = relationships.files[INDEX]
    relationships_name = 'airplane_receptacle_aabb_TvStnd1_Top3_frl_apartment_tvstand_all_0'  # hard code
    choose_relationship = relationships[relationships_name]
    obj_offset = choose_relationship[0]
    obj_rot_mat = choose_relationship[1]
    holder_idx = choose_relationship[2]
    print(f'relationship name: {relationships_name}, obj_offset: {obj_offset.shape}, obj_rot_mat: {obj_rot_mat.shape}, holder_idx: {holder_idx}')

    holder = np.load(holder_path, allow_pickle=True)
    holder_name = '_'.join(relationships_name.split('_')[1:-2])
    # assert holder_name == holder.files[holder_idx]
    holder_mesh = holder[holder_name]
    holder_vertices = holder_mesh[0][0]
    holder_faces = holder_mesh[0][1]
    print(f'holder name: {holder_name}, vertice: {holder_mesh[0][0].shape}, face: {holder_mesh[0][1].shape}')

    obj_name = relationships_name.split('_')[0]
    obj_mesh = trimesh.load_mesh(f'{obj_mesh_root}/{obj_name}.ply')
    # obj_bps = np.load(obj_bps_path, allow_pickle=True)
    # print(f'obj_bps files: {obj_bps.files}')
    # obj_bps = obj_bps[obj_name]
    # obj_info = np.load(obj_info_path, allow_pickle=True)
    obj_vertices = obj_mesh.vertices
    obj_faces = obj_mesh.faces
    print(f'obj_name: {obj_name}, vertice: {obj_vertices.shape}, face: {obj_faces.shape}')

    rr.log(f'{relationships_name}/{holder_name}',
           rr.Mesh3D(vertex_positions=holder_vertices,
                     triangle_indices=holder_faces,
                     vertex_normals=compute_vertex_normals(holder_vertices, holder_faces)))

    rot = R.from_matrix(obj_rot_mat)
    pos = obj_offset
    obj_vertices = rot.apply(obj_vertices) + pos
    rr.log(f'{relationships_name}/{obj_name}',
           rr.Mesh3D(vertex_positions=obj_vertices,
                     triangle_indices=obj_faces,
                     vertex_normals=compute_vertex_normals(obj_vertices, obj_faces)))


def write_human():
    human_params = dict(np.load(motion_path, allow_pickle=True))
    human_params = (human_params['arr_0'].item())['final_results']
    human_model = smplx.create(model_path=config.SMPL_MODEL_PATH,
                               model_type='smplx',
                               gender='neutral',
                               num_pca_comps=np.array(24),
                               batch_size=1).to('cpu').eval()
    for idx in range(5):
        res_i = human_params[idx]
        translation = res_i['transl'].reshape(1, 3)
        orientation = R.from_matrix(res_i['global_orient']).as_rotvec().reshape(1, 3)
        pose = res_i['pose'].reshape(1, 63)

        output = human_model(body_pose=torch.tensor(pose, dtype=torch.float32),
                             global_orient=torch.tensor(orientation, dtype=torch.float32),
                             transl=torch.tensor(translation, dtype=torch.float32))

        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = human_model.faces
        rr.log(f'human/{idx}',
               rr.Mesh3D(vertex_positions=vertices,
                         triangle_indices=faces,
                         vertex_normals=compute_vertex_normals(vertices, faces),))


if __name__ == '__main__':
    set_up_rerun()
    write_obj()
    write_human()
