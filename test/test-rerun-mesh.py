import trimesh
import rerun as rr

# 创建一个示例的trimesh对象
mesh = trimesh.Trimesh(vertices=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       faces=[[0, 1, 2]])

# 提取顶点位置和面信息
vertex_positions = mesh.vertices.tolist()
triangle_indices = mesh.faces.flatten().tolist()

# 假设所有法线相同，这里只是一个示例
vertex_normals = [[0.0, 0.0, 1.0]] * len(vertex_positions)

# 假设所有顶点颜色不同，这里只是一个示例
vertex_colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]

# 使用你提供的格式记录网格
rr.init("rerun_example_mesh", spawn=True)
rr.log(
    "triangle",
    rr.Mesh3D(
        vertex_positions=vertex_positions,
        triangle_indices=triangle_indices,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors
    ),
)
