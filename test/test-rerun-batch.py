import numpy as np
import rerun as rr

rr.init("rerun_example_send_columns", spawn=True)

times = np.arange(0, 64)
scalars = np.sin(times / 10.0)

# Send both columns in a single call.
rr.send_columns(
    "scalars",
    times=[rr.TimeSequenceColumn("step", times)],
    components=[rr.components.ScalarBatch(scalars)],
)
