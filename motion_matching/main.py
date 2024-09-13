import numpy as np
import rerun as rr
import smplx
import torch
import config
from utils.mesh_utils import compute_vertex_normals

def main():
    # Load SFU MoCap data from NPZ file
    bdata = np.load('/Users/emptyblue/Documents/Research/layout_design/motion_matching/AMASS/SFU_mocap/0005_Walking001_stageii.npz')
    print(bdata.files)
    
    poses = bdata['poses']
    poses = poses[:100].reshape(100, -1, 3)
    print(poses.shape)
    
    time_length = len(bdata['trans'])

    comp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    body_parms = {
        'global_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'body_pose': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'hand_pose': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'transl': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        # 'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
    }

    # Limit to 20 frames
    max_frames = min(20, time_length)
    
    # Adjust body parameters to use only max_frames
    body_parms = {k: v[:max_frames] for k, v in body_parms.items()}
    
    # Update time_length
    time_length = max_frames
    
    print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    print('time_length = {}'.format(time_length))
    
    rr.init("sfu_mocap_visualization", spawn=True)
    # rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)

    bm = smplx.create(model_path=config.SMPL_MODEL_PATH,
                      model_type='smplx',
                      gender='neutral',
                      use_face_contour=False,
                      num_betas=num_betas,
                      num_dmpls=num_dmpls,
                      ext='npz',
                      batch_size=time_length)
    faces = bm.faces

    # Create the body model once
    body = bm(**{k: v.to(comp_device) for k, v in body_parms.items()})
    
    for frame in range(time_length):
        vertices = body.vertices[frame].detach().cpu().numpy()
        
        rr.set_time_seconds("stable_time", frame / 30.0)  # Assuming 30 fps
        rr.log("human", 
               rr.Mesh3D(
                   vertex_positions=vertices,
                   triangle_indices=faces,
                   vertex_normals=compute_vertex_normals(vertices, faces)
               ))
        


if __name__ == "__main__":
    main()