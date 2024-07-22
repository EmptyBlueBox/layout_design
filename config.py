import torch
import os

DATASET_PATH = '/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS'
OBJECT_ORIGINAL_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_mesh')
OBJECT_DECIMATED_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_mesh_decimated')
OBJECT_SDF_GRID_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_SDF_grid')
OBJECT_SDF_INFO_PATH = os.path.join(DATASET_PATH, 'Object_all/Object_SDF_info')


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


device = setup_device()
