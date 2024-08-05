from rerun.datatypes import Angle, RotationAxisAngle
from math import pi
import rerun as rr

rr.init("object_with_mtl")
rr.save("test_mtl.rrd")
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis

rr_transform = rr.Transform3D(
    translation=[0, 1, 0],
    rotation=RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=pi / 4)),
    scale=0.5,
)
rr.log("world/asset", rr_transform)
rr.log("world/asset", rr.Asset3D(path='/Users/emptyblue/Documents/Research/layout_design/test/test-3DFRONT/0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e/raw_model.obj'))
