import numpy as np


def trilinear_interpolation_vectorized(data, queries):
    """
    三维双线性插值的向量化实现
    :param data: 原始数据, 形状为(a, b, c)
    :param queries: 查询点, 形状为(n, 3)，每个查询点在[0, 1]之间
    :return: 插值后的值, 形状为(n,)
    """
    a, b, c = data.shape
    n = queries.shape[0]

    # 将查询点的比例转换为实际坐标
    queries_scaled = queries * [a - 1, b - 1, c - 1]

    # 获取查询点的整数和小数部分
    queries_int = np.floor(queries_scaled).astype(int)
    queries_frac = queries_scaled - queries_int

    # 获取8个顶点的坐标
    x0, y0, z0 = queries_int[:, 0], queries_int[:, 1], queries_int[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # 边界检查，防止索引超出范围
    x1 = np.clip(x1, 0, a - 1)
    y1 = np.clip(y1, 0, b - 1)
    z1 = np.clip(z1, 0, c - 1)

    # 获取8个顶点的值
    f000 = data[x0, y0, z0]
    f100 = data[x1, y0, z0]
    f010 = data[x0, y1, z0]
    f110 = data[x1, y1, z0]
    f001 = data[x0, y0, z1]
    f101 = data[x1, y0, z1]
    f011 = data[x0, y1, z1]
    f111 = data[x1, y1, z1]

    # 在x方向进行线性插值
    fx00 = f000 * (1 - queries_frac[:, 0]) + f100 * queries_frac[:, 0]
    fx10 = f010 * (1 - queries_frac[:, 0]) + f110 * queries_frac[:, 0]
    fx01 = f001 * (1 - queries_frac[:, 0]) + f101 * queries_frac[:, 0]
    fx11 = f011 * (1 - queries_frac[:, 0]) + f111 * queries_frac[:, 0]

    # 在y方向进行线性插值
    fxy0 = fx00 * (1 - queries_frac[:, 1]) + fx10 * queries_frac[:, 1]
    fxy1 = fx01 * (1 - queries_frac[:, 1]) + fx11 * queries_frac[:, 1]

    # 在z方向进行线性插值
    fxyz = fxy0 * (1 - queries_frac[:, 2]) + fxy1 * queries_frac[:, 2]

    return fxyz


# 示例数据
data = np.arange(27).reshape(3, 3, 3)
print(data)

# 查询点
queries = np.array([
    [0.5, 0.5, 0.5],
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7]
])

# 插值计算
result = trilinear_interpolation_vectorized(data, queries)
print(result)  # 输出插值结果
