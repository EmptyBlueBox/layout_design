import torch
import os


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


device = setup_device()

current_file_path = os.path.dirname(os.path.abspath(__file__))  # '/Users/emptyblue/Documents/Research/layout_design/'

SMPL_MODEL_PATH = '/Users/emptyblue/Documents/Research/HUMAN_MODELS'

# SHADE
DATASET_SHADE_PATH = os.path.join(current_file_path, 'dataset', 'SHADE')

# TRUMANS
DATASET_TRUMANS_PATH = os.path.join(current_file_path, 'dataset', 'TRUMANS')
OBJECT_ORIGINAL_PATH = os.path.join(DATASET_TRUMANS_PATH, 'Object_all', 'Object_mesh')
OBJECT_DECIMATED_PATH = os.path.join(DATASET_TRUMANS_PATH, 'Object_all', 'Object_mesh_decimated')
OBJECT_SDF_GRID_PATH = os.path.join(DATASET_TRUMANS_PATH, 'Object_all', 'Object_SDF_grid')
OBJECT_SDF_INFO_PATH = os.path.join(DATASET_TRUMANS_PATH, 'Object_all', 'Object_SDF_info')

# 3D-FRONT
DATASET_3DFRONT_ROOT_PATH = '/viscam/data/3D_FRONT_FUTURE/3D_FRONT'
DATASET_3DFRONT_LAYOUT_PATH = os.path.join(DATASET_3DFRONT_ROOT_PATH, '3D-FRONT')
DATASET_3DFRONT_TEXTURE_PATH = os.path.join(DATASET_3DFRONT_ROOT_PATH, '3D-FRONT-texture')
DATASET_3DFUTURE_MODEL_PATH = os.path.join(DATASET_3DFRONT_ROOT_PATH, '3D-FUTURE-model')
DATASET_3DFUTURE_MODEL_INFO_PATH = os.path.join(DATASET_3DFUTURE_MODEL_PATH, 'model_info.json')

# HOLODECK
DATA_HOLODECK_PATH = os.path.join(current_file_path, 'dataset', 'HOLODECK')

# OBJATHOR
OBJATHOR_BASE = '/Users/emptyblue/.objathor-assets/2023_09_23/assets/'
DATA_OBJATHOR_CACHE_PATH = os.path.join(current_file_path, 'dataset', 'OBJATHOR', 'object_inner_points_cache')

ROOM_SHAPE_X = 8
ROOM_SHAPE_Z = 8
