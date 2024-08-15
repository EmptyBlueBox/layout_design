'''
可视化背景数据, 包含墙壁和物体
背景数据: 
一个 (300, 100, 400) 三维数组, 每个元素的值为0或1, 0表示背景, 1表示物体
50个点代表1米
'''
import rerun as rr
import numpy as np
import os
import config

background_path = os.path.join(config.DATASET_TRUMANS_PATH, 'background', 'background.npy')  # (300, 100, 400)

HOLODECK_NAME = 'a_DiningRoom_with_round_table_-2024-08-07-14-52-49-547177'
background_path = os.path.join(config.DATA_HOLODECK_PATH, HOLODECK_NAME, 'background.npy')  # (300, 100, 400)

save = False


def set_up_rerun():
    # start rerun script
    rr.init('Visualization: bg', spawn=not save)
    if save:
        rr.save(os.path.join('rerun', 'TRUMANS_bg.rrd'))
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # Set an up-axis = +Y
    rr.set_time_seconds("stable_time", 0)


def downsample_background(background, factor=10):
    """
    对背景数据进行降采样。
    """
    # 获取背景数据的尺寸
    dim_x, dim_y, dim_z = background.shape

    # 计算降采样后的尺寸
    new_dim_x = dim_x // factor
    new_dim_y = dim_y // factor
    new_dim_z = dim_z // factor

    # 使用reshape和mean进行降采样
    downsampled_background = background.reshape(new_dim_x, factor, new_dim_y, factor, new_dim_z, factor).mean(axis=(1, 3, 5))

    return downsampled_background


def calculate_coordinates_and_colors(background, grid_size=(6, 2, 8)):
    """
    计算每个点的坐标和颜色。
    """
    # 确定原始数组的尺寸
    dim_x, dim_y, dim_z = background.shape

    # 计算每个点的坐标
    x_coords = np.linspace(0, grid_size[0], dim_x, endpoint=False)
    y_coords = np.linspace(0, grid_size[1], dim_y, endpoint=False)
    z_coords = np.linspace(0, grid_size[2], dim_z, endpoint=False)

    # 使用meshgrid生成坐标矩阵
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # 将坐标矩阵展开为N*3的数组
    coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # 生成颜色矩阵
    colors = np.where(background.ravel()[:, np.newaxis] == 1, [255, 0, 0], [0, 255, 0])

    return coordinates, colors


def test_original_background(grid_size=(6, 2, 8), downsample_factor=10):
    """
    测试函数：加载背景数据，降采样，计算坐标和颜色。
    """
    background = np.load(background_path)

    # 根据grid_size裁剪background
    points_per_meter = 50
    x_size = int(grid_size[0] * points_per_meter)
    y_size = int(grid_size[1] * points_per_meter)
    z_size = int(grid_size[2] * points_per_meter)
    background = background[:x_size, :y_size, :z_size]
    # background = background[200:x_size, :y_size, :150]

    # 先降采样
    downsampled_background = downsample_background(background, factor=downsample_factor)

    # 计算每个点的坐标和颜色
    bg_coor, bg_color = calculate_coordinates_and_colors(downsampled_background, grid_size)

    red_mask = np.all(bg_color == [255, 0, 0], axis=1)
    # red_mask = np.any(bg_color != [0, 255, 0], axis=1)

    # 过滤出红色的点
    bg_coor = bg_coor[red_mask, :]
    bg_color = bg_color[red_mask, :]

    # 输出结果
    print("Coordinates shape:", bg_coor.shape)
    print("Colors shape:", bg_color.shape)

    rr.log('point_cloud', rr.Points3D(positions=bg_coor, colors=bg_color))

    return


def main():
    set_up_rerun()
    test_original_background((6, 2, 8), 1)

    # chair = np.load('../dataset/TRUMANS/background/kitchen_chair_1.npy')
    # rr.log('chair', rr.Points3D(chair))


if __name__ == '__main__':
    main()
