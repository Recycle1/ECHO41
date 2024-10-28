from torch.utils.data import Dataset
import torch

class AudioVisualDataset(Dataset):
    def __init__(self, video_features, audio_mels):
        assert len(video_features) == len(audio_mels), "视频和音频样本数量不一致"
        self.video_features = video_features
        self.audio_mels = audio_mels

    def __len__(self):
        return len(self.audio_mels)

    def __getitem__(self, idx):
        video_feature = self.video_features[idx]   # [context_length, feature_dim]
        audio_mel = self.audio_mels[idx]        # [1, freq_bins, time_steps]

        # 确保数据是 FloatTensor
        video_feature = video_feature.float()
        audio_mel = audio_mel.float()

        return audio_mel, video_feature

def extract_features(mel_spec_tensor, audio_model):
    # audio_model.half()
    mel_spec_tensor = mel_spec_tensor.squeeze(1)
    print(mel_spec_tensor.shape)
    with autocast('cuda'):
        features = audio_model(mel_spec_tensor)

    return features.last_hidden_state

# 加载视频特征
video_features = torch.load('videos.pt', weights_only=True)  # 确保安全加载权重
audio_mels = torch.load('audios.pt', weights_only=True)

print(video_features.shape)
print(audio_mels.shape)

from torch.utils.data import DataLoader

# 创建数据集
dataset = AudioVisualDataset(video_features, audio_mels)

# 创建数据加载器
batch_size = 4 # 根据显存大小调整
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 假设 UNet 已经定义，如之前的回答
from ModelZoo.ECHO41 import UNet

# 实例化模型
model = UNet(
    in_channels=1,
    out_channels_final=1,
    base_channels=64,
    channel_mults=(1, 2, 4, 8, 8),
    num_res_blocks=2,
    time_emb_dim=256,
    num_heads=8
)

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

import torch.optim as optim

# 设置优化器
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

import torch

# 定义噪声调度参数
num_timesteps = 1000  # 总的扩散步骤数

beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)  # [T]

# 计算 alpha 和累计乘积
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]

# 提前计算 sqrt(alpha_cumprod) 和 sqrt(1 - alpha_cumprod)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)            # [T]
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)  # [T]

import tqdm  # 用于显示训练进度

num_epochs = 15  # 训练轮数

from torch.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (x_0, v) in progress_bar:
        x_0 = x_0.to(device)  # [Batch, Freq, Time]
        if x_0.ndim == 3:
            x_0 = x_0.unsqueeze(1)  # 添加通道维度，形状变为 [Batch, 1, Freq, Time]
        v = v.to(device)  # [Batch, Seq Len, Feature Dim]
        batch_size = x_0.size(0)

        # 对视频特征进行下采样，减少序列长度
        stride = 2  # 根据需要调整
        v = v[:, ::stride, :]  # [Batch, Seq Len/stride, Feature Dim]

        # 随机采样时间步 t
        t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()  # [Batch]

        # 获取对应时间步的 sqrt_alphas_cumprod 和 sqrt_one_minus_alphas_cumprod
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)  # [Batch, 1, 1, 1]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)  # [Batch, 1, 1, 1]

        # 采样噪声 epsilon
        epsilon = torch.randn_like(x_0).to(device)

        # 生成带噪声的 x_t
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * epsilon

        # 前向传播和损失计算
        with autocast(device_type='cuda'):
            criterion = torch.nn.MSELoss()
            epsilon_theta = model(x_t, t, v)  # [Batch, 1, Freq, Time]
            loss = criterion(epsilon_theta, epsilon)
            # loss = loss_f(epsilon_theta, epsilon)

        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * batch_size

        # 更新进度条
        progress_bar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        progress_bar.set_postfix(loss=loss.item())

    # 打印每个 epoch 的平均损失
    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), 'unet_model.pth')



