"""Log a simple 3D asset."""

from rerun.datatypes import Angle, RotationAxisAngle
from math import pi
import sys

import rerun as rr

path = '../dataset/object_obj/box/box_face1000.obj'

rr.init("rerun_example_asset3d", spawn=True)

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
rr.log("world/asset1", rr.Asset3D(path=path))

rr_transform = rr.Transform3D(
    rotation=RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=pi / 4)),
    scale=2,
)
rr.log("world/asset2", rr.Transform3D(translation=[1, 0, 0]))
rr.log("world/asset2", rr.Asset3D(path=path))


rr.init("rerun_example_transform3d", spawn=True)

arrow = rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 1, 0])

rr.log("base", arrow)

rr.log(
    "base/rotated_scaled1",
    rr.Transform3D(
        translation=[0, 0, 1],
        rotation=RotationAxisAngle(axis=[0, 1, 0], angle=Angle(rad=pi / 4)),
        scale=2,
    ),
)
rr.log("base/rotated_scaled1", arrow)
