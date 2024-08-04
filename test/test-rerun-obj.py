"""Log a simple 3D asset."""

from rerun.datatypes import Angle, RotationAxisAngle
from math import pi
import sys

import rerun as rr

# path = '/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS/Object_all/Object_mesh/cup_01.obj'

# rr.init("rerun_example_asset3d", spawn=True)

# rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
# rr.log("world/asset1", rr.Asset3D(path=path))

# rr_transform = rr.Transform3D(
#     rotation=RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=pi / 4)),
#     scale=2,
# )
# rr.log("world/asset2", rr.Transform3D(translation=[1, 0, 0]))
# rr.log("world/asset2", rr.Asset3D(path=path))


rr.init("rerun_example_transform3d")
rr.save("test.rrd")

rr.log("simple1", rr.Boxes3D(half_sizes=[2.0, 2.0, 1.0]))
rr.log("simple2", rr.Boxes3D(half_sizes=[1.0, 1.0, 1.0]))
