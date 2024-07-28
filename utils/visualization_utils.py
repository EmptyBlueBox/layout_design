import smplx
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from rerun.datatypes import Quaternion, Angle, RotationAxisAngle
import rerun as rr
import trimesh
import pickle
import torch
from utils.mesh_utils import compute_vertex_normals, farthest_point_sampling
from utils.pytorch3d_utils import quaternion_multiply, axis_angle_to_quaternion, quaternion_to_axis_angle, quaternion_apply
import config
import os


