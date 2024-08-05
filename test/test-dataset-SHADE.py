import pandas as pd
import os
import pickle
import config

dataset_path = config.DATASET_SHADE_PATH
living_room_test_path = os.path.join(dataset_path, 'livingroom', 'calibrated_livingroom_test.pkl')
living_room_test_path = os.path.join(dataset_path, 'bedroom', 'calibrated_bedroom_test.pkl')

with open(living_room_test_path, 'rb') as f:
    living_room_test = pickle.load(f)
print(living_room_test.keys())


# # 定义加载数据集的函数

# def load_dataset(file_path):
#     try:
#         # 使用Pandas加载.pkl文件
#         dataset = pd.read_pickle(file_path)
#         return dataset
#     except FileNotFoundError:
#         print(f"文件 {file_path} 不存在。")
#     except Exception as e:
#         print(f"加载文件 {file_path} 时发生错误: {e}")


# # 使用示例
# dataset = load_dataset(living_room_test_path)

# # 打印数据集的类型和一些内容以确保加载正确
# print(type(dataset))
# print(dataset)
