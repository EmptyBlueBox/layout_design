import os
import subprocess

# # 设置源文件夹和目标文件夹路径
# source_folder = "/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS/Object_all/Object_mesh"
# target_folder = "/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS/Object_all/Object_mesh_decimated"

# # 检查目标文件夹是否存在，如果不存在则创建
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# # 遍历源文件夹中的所有文件
# for filename in os.listdir(source_folder):
#     # 检查文件是否为 .obj 文件
#     if filename.endswith(".obj"):
#         # 构造源文件和目标文件的完整路径
#         source_file = os.path.join(source_folder, filename)
#         target_file = os.path.join(target_folder, filename)

#         # 构造 simplify_obj 命令
#         command = ["./simplify_obj", source_file, target_file, "0.2"]

#         # 打印命令以供调试
#         print("Executing command:", " ".join(command))

#         # 执行命令
#         try:
#             subprocess.run(command, check=True)
#             print(f"Successfully simplified {filename}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error simplifying {filename}: {e}")

source_folder = "/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS/Scene_mesh"
target_folder = "/Users/emptyblue/Documents/Research/layout_design/dataset/TRUMANS/Scene_mesh_decimated"

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 检查文件是否为 .obj 文件
    if filename.endswith(".obj"):
        # 构造源文件和目标文件的完整路径
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)

        # 构造 simplify_obj 命令
        command = ["./simplify_obj", source_file, target_file, "0.1"]

        # 打印命令以供调试
        print("Executing command:", " ".join(command))

        # 执行命令
        try:
            subprocess.run(command, check=True)
            print(f"Successfully simplified {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error simplifying {filename}: {e}")
