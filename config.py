import torch
import os

DATASET_SHADE_PATH = '/Users/emptyblue/Documents/Research/layout_design/dataset/SHADE'

DATASET_PATH = '/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS'
OBJECT_ORIGINAL_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_mesh')
OBJECT_DECIMATED_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_mesh_decimated')
OBJECT_SDF_GRID_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_SDF_grid')
OBJECT_SDF_INFO_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_SDF_info')


def setup_device():
    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # 设置当前设备为第一个 GPU
        torch.cuda.set_device(device)
    else:
        # 如果没有可用的 GPU，则使用 CPU
        device = torch.device("cpu")
    return device


device = setup_device()

ROOM_SHAPE_X = 6
ROOM_SHAPE_Z = 6

SMPL_MODEL_PATH = '/Users/emptyblue/Documents/Research/HUMAN_MODELS'
