import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)

robot_id = p.loadURDF("../humanoid/humanoid.urdf")

# 设置初始仿真参数
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(1)

# 让仿真运行一段时间，以便观察
try:
    while True:
        time.sleep(1./240.)  # 控制仿真速度
except KeyboardInterrupt:
    # 按Ctrl+C退出仿真
    pass

# 断开连接
p.disconnect()
