import os
import librosa
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_audio_to_mel_spec(audio_path, target_sr=16000, n_fft=1024, hop_length=512, n_mels=80):
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None)

    # 采样率调整
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    # 转换为对数梅尔频谱
    log_mel_spec = librosa.power_to_db(mel_spec, ref=20)

    return log_mel_spec

# 定义源文件夹和目标文件路径
source_folder = "./Data/train/Audios/Race car, auto racing"
target_file = "audios.pt"

# 初始化一个空列表用于存储所有数据
all_log_mel_specs = []

# 第一次遍历源文件夹，计算全局均值和标准差
for filename in os.listdir(source_folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(source_folder, filename)
        # 加载音频并计算对数梅尔谱
        log_mel_spec = load_audio_to_mel_spec(file_path)
        all_log_mel_specs.append(log_mel_spec)

# 将所有log_mel_specs合并为一个大数组
all_log_mel_specs_combined = np.concatenate(all_log_mel_specs, axis=1)
global_mean = all_log_mel_specs_combined.mean()
global_std = all_log_mel_specs_combined.std()

print(f"全局均值: {global_mean}, 全局标准差: {global_std}")

# 第二次遍历源文件夹，对每个文件进行标准化并保存
all_data = []

for filename in os.listdir(source_folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(source_folder, filename)
        # 加载音频并计算对数梅尔谱
        log_mel_spec = load_audio_to_mel_spec(file_path)
        # 使用全局均值和标准差进行归一化
        log_mel_spec_norm = (log_mel_spec - global_mean) / global_std
        # 转换为 Tensor 并调整维度
        mel_spec_tensor = torch.tensor(log_mel_spec_norm).unsqueeze(0)  # [1, n_mels, T]
        # 添加到列表中
        all_data.append(mel_spec_tensor)

# 将所有数据合并为一个大的 Tensor
combined_data = torch.cat(all_data, dim=0)

# 保存为 .pt 文件
torch.save(combined_data, target_file)

print(f"合并后的数据已保存到 {target_file}")