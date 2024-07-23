import pybullet as p

# 连接pybullet物理引擎
physicsClient = p.connect(p.GUI)

# 加载细棒模型
visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 5], rgbaColor=[1, 0, 0, 1])
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 5])
barId = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=[0, 0, 0])

# 设置细棒的初始速度和角速度
p.resetBaseVelocity(barId, linearVelocity=[1, 0, 0], angularVelocity=[0, 1, 0])

# 获取细棒的速度和质量
linearVel, angularVel = p.getBaseVelocity(barId)
mass = p.getDynamicsInfo(barId, -1)[0]

# 计算动能
kineticEnergy = 0.5 * mass * (linearVel[0]**2 + linearVel[1]**2 + linearVel[2]**2)

# 计算转动能量
inertia = p.getDynamicsInfo(barId, -1)[2]
angularKineticEnergy = 0.5 * sum(inertia[i] * angularVel[i]**2 for i in range(3))

totalKineticEnergy = kineticEnergy + angularKineticEnergy

print(f'Bar size: {p.getVisualShapeData(barId)[0][3]}')
print("Bar linear velocity: ", linearVel)
print("Bar angular velocity: ", angularVel)
print("Bar mass: ", mass)
print("Bar inertia: ", inertia)
print("Bar kinetic energy: ", kineticEnergy)
print("Bar angular kinetic energy: ", angularKineticEnergy)

# 断开物理引擎连接
p.disconnect()
