import rerun as rr
import numpy as np
import smplx
import config
import torch
from utils.mesh_utils import compute_vertex_normals
from MotionMatching.main import main
import os
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

down_sample_rate = 5
interpolation_frames = 100  # 超参数，控制lerp和slerp在多少帧以内完成
touch_threshold = 0.6  # 设置一个合适的阈值

keypoints_good_added = [
    [],
    [[4.5,1.5]],
    [],
    [],
    [[4.5,3]],
    [],
    [],
    []
]

keypoints_bad_added = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    []
]

def lerp(a, b, t):
    # Ensure a and b are the same shape
    a, b = np.broadcast_arrays(a, b)
    return a + t * (b - a)


def slerp(a, b, t):
    a_rot = R.from_rotvec(a)
    b_rot = R.from_rotvec(b)
    slerp_obj = Slerp([0, 1], R.from_rotvec([a, b]))  # Corrected this line
    return slerp_obj(t).as_rotvec()

def interpolate_motion(motion, start_pose, end_pose, interpolation_frames):
    start_translation = start_pose["translation"]
    end_translation = end_pose["translation"]
    start_orientation = start_pose["orientation"]
    end_orientation = end_pose["orientation"]
    start_body_pose = start_pose["body_pose"]
    end_body_pose = end_pose["body_pose"]

    if motion["translation"].shape[0] < interpolation_frames:
        return motion  # Skip interpolation if not enough frames

    for t in range(1, interpolation_frames + 1):
        alpha = t / (interpolation_frames + 1)
        motion["translation"][-t] = lerp(end_translation, motion["translation"][-t], alpha)
        motion["translation"][t - 1] = lerp(start_translation, motion["translation"][t - 1], alpha)
        motion["orientation"][-t] = slerp(end_orientation, motion["orientation"][-t], alpha)
        motion["orientation"][t - 1] = slerp(start_orientation, motion["orientation"][t - 1], alpha)
        motion["body_pose"][-t] = lerp(end_body_pose, motion["body_pose"][-t], alpha)
        motion["body_pose"][t - 1] = lerp(start_body_pose, motion["body_pose"][t - 1], alpha)

    return motion

def viz_motion_matching(motion_matching_human_params, type='good'):    
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
    
    # # key frames
    # for i in range(len(motion_matching_human_params["transl"])):
    #     output = human_model(
    #         body_pose=torch.tensor(np.array(motion_matching_human_params["body_pose"][i]).reshape(1, 63), dtype=torch.float32),
    #         global_orient=torch.tensor(np.array(motion_matching_human_params["global_orient"][i]).reshape(1, 3), dtype=torch.float32),
    #         transl=torch.tensor(np.array(motion_matching_human_params["transl"][i]).reshape(1, 3), dtype=torch.float32),
    #     )
    #     vertices = output.vertices.detach().cpu().numpy()[0]
    #     faces = human_model.faces
    #     rr.set_time_seconds("stable_time", i)
    #     rr.log(
    #         "human",
    #         rr.Mesh3D(
    #             vertex_positions=vertices,
    #             triangle_indices=faces,
    #             vertex_normals=compute_vertex_normals(vertices, faces),
    #         )) 
    
    master_keypoints = np.array(motion_matching_human_params["transl"])[:,[0,2]]
    original_len=len(master_keypoints)
    
    root_positions = []

    current_time = 0  # Initialize current time

    for i in range(original_len - 1):
        start_keypoint = master_keypoints[i]
        end_keypoint = master_keypoints[i + 1]
        
        distance = np.linalg.norm(start_keypoint - end_keypoint)
        
        if distance < touch_threshold:
            # Use original 3D transl keypoints
            start_keypoint_3d = motion_matching_human_params["transl"][i]
            end_keypoint_3d = motion_matching_human_params["transl"][i + 1]
            start_orientation = motion_matching_human_params["global_orient"][i]
            end_orientation = motion_matching_human_params["global_orient"][i + 1]
            
            # Create a motion sequence with interpolation_frames length
            motion = {
                "translation": np.array([lerp(start_keypoint_3d, end_keypoint_3d, t / (interpolation_frames + 1)) for t in range(1, interpolation_frames + 1)]),
                "orientation": np.array([slerp(start_orientation, end_orientation, t / (interpolation_frames + 1)) for t in range(1, interpolation_frames + 1)]),
                "body_pose": np.zeros((interpolation_frames, 63)),  # Assuming 63D body pose
            }
        else:
            keypoints=np.array([start_keypoint, end_keypoint])
            if type == 'good':
                if len(keypoints_good_added[i]) > 0:
                    keypoints=np.insert(keypoints, 1, keypoints_good_added[i], axis=0)
            elif type == 'bad':
                if len(keypoints_bad_added[i]) > 0:
                    keypoints=np.insert(keypoints, 1, keypoints_bad_added[i], axis=0)
            else:
                print(f'type error')
                exit()
            print(f'keypoints: {keypoints}')
            motion = main(keypoints=keypoints, threshold=touch_threshold)
            motion["translation"] = motion["translation"]+np.array([0, 0.36, 0]) # 添加一个偏移量，使得在视觉上更合理
            
            start_keypoint_3d = motion_matching_human_params["transl"][i]
            end_keypoint_3d = motion_matching_human_params["transl"][i + 1]
            start_orientation = motion_matching_human_params["global_orient"][i]
            end_orientation = motion_matching_human_params["global_orient"][i + 1]
        
        if len(motion["translation"]) == 0:
            # Create a motion sequence with interpolation_frames length
            motion["translation"] = np.array([lerp(start_keypoint, end_keypoint, t / (interpolation_frames + 1)) for t in range(1, interpolation_frames + 1)])
            motion["orientation"] = np.zeros((interpolation_frames, 3))  # Assuming 3D orientation
            motion["body_pose"] = np.zeros((interpolation_frames, 63))  # Assuming 63D body pose

        # Interpolate the motion    
        motion = interpolate_motion(motion, 
                                    {"translation": start_keypoint_3d, "orientation": start_orientation, "body_pose": motion_matching_human_params["body_pose"][i]},
                                    {"translation": end_keypoint_3d, "orientation": end_orientation, "body_pose": motion_matching_human_params["body_pose"][i + 1]},
                                    interpolation_frames)

        # Downsample the motion
        motion["translation"] = motion["translation"][::down_sample_rate]
        motion["orientation"] = motion["orientation"][::down_sample_rate]
        motion["body_pose"] = motion["body_pose"][::down_sample_rate]
        
        smplx_model = smplx.create(model_path=os.environ.get("SMPL_MODEL_PATH"),
                          model_type='smplx',
                          gender='neutral',
                          use_face_contour=False,
                          ext='npz',
                          batch_size=motion["translation"].shape[0])
        
        body_parms = {
            'global_orient': torch.Tensor(motion["orientation"]), # controls the global root orientation
            'body_pose': torch.Tensor(motion["body_pose"]), # controls the body
            'transl': torch.Tensor(motion["translation"]).reshape(-1, 3), # Ensure correct shape
        }
        smplx_output = smplx_model(**{k: v for k, v in body_parms.items()})
        
        # 可视化 start pose
        output = human_model(
            body_pose=torch.tensor(np.array(motion_matching_human_params["body_pose"][i]).reshape(1, 63), dtype=torch.float32),
            global_orient=torch.tensor(np.array(motion_matching_human_params["global_orient"][i]).reshape(1, 3), dtype=torch.float32),
            transl=torch.tensor(np.array(motion_matching_human_params["transl"][i]).reshape(1, 3), dtype=torch.float32),
        )
        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = human_model.faces
        rr.set_time_seconds("stable_time", current_time)
        rr.log(
            "human_start",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=compute_vertex_normals(vertices, faces),
            )) 
        # 可视化 end pose
        output = human_model(
            body_pose=torch.tensor(np.array(motion_matching_human_params["body_pose"][i + 1]).reshape(1, 63), dtype=torch.float32),
            global_orient=torch.tensor(np.array(motion_matching_human_params["global_orient"][i + 1]).reshape(1, 3), dtype=torch.float32),
            transl=torch.tensor(np.array(motion_matching_human_params["transl"][i + 1]).reshape(1, 3), dtype=torch.float32),
        )
        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = human_model.faces
        rr.set_time_seconds("stable_time", current_time)
        rr.log(
            "human_end",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=compute_vertex_normals(vertices, faces),
            )) 
        
        # 可视化 motion
        for frame in range(motion["translation"].shape[0]):
            vertices = smplx_output.vertices[frame].detach().cpu().numpy()
            joints = smplx_output.joints[frame].detach().cpu().numpy()
            
            # Record root position
            root_positions.append(motion["translation"][frame])

            rr.set_time_seconds("stable_time", current_time)  # Set time for each frame
            rr.log("motion", 
                   rr.Mesh3D(
                       vertex_positions=vertices,
                       triangle_indices=smplx_model.faces,
                       vertex_normals=compute_vertex_normals(vertices, smplx_model.faces)
                   ))
            current_time += 1 / 120.0 * down_sample_rate  # Increment time
            

    # Convert root positions to numpy array for visualization
    root_positions = np.array(root_positions)

    # Visualize root trajectory
    rr.log("root_trajectory", 
           rr.Points3D(
               positions=root_positions,
               colors=np.full(root_positions.shape, 255, dtype=np.uint8),
           ))