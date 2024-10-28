import os
import torch

# 定义源文件夹和目标文件路径
source_folder = "./Data/Processed/Video(only)/Race car, auto racing"
target_file = "videos.pt"

# 初始化一个空列表用于存储所有数据
all_data = []

# 遍历源文件夹中的所有 .pt 文件
for filename in os.listdir(source_folder):
    if filename.endswith(".pt"):
        file_path = os.path.join(source_folder, filename)
        # 加载 .pt 文件
        data = torch.load(file_path)
        # 添加到列表中
        all_data.append(data.unsqueeze(0))  # 添加一个新的维度以保持 [1, ...] 形式

# 将所有数据合并为一个大的 Tensor，形状为 [52, ...]
combined_data = torch.cat(all_data, dim=0)

# 保存为 .pt 文件
torch.save(combined_data, target_file)

print(f"合并后的数据已保存到 {target_file}")
