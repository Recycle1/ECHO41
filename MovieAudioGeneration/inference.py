from ModelZoo.ECHO41 import UNet
import torch
import librosa.display
import soundfile as sf
from torch.amp import autocast
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(
    in_channels=1,
    out_channels_final=1,
    base_channels=64,
    channel_mults=(1, 2, 4, 8, 8),
    num_res_blocks=2,
    time_emb_dim=256,
    num_heads=8
)

model.load_state_dict(torch.load('unet_model.pth', weights_only=True))
model.to(device)

num_timesteps = 1000

beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)  # [T]

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]

# 提前计算 sqrt(alpha_cumprod) 和 sqrt(1 - alpha_cumprod)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)            # [T]
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)  # [T]
freq_bins = 80
time_steps = 313

def sample(model, v, num_timesteps, betas, device, freq_bins, time_steps):
    model.eval()
    with torch.no_grad():
        batch_size = v.size(0)
        x_t = torch.randn(batch_size, 1, freq_bins, time_steps, device=device)  # 初始噪声
        # print(f"x_t initialized: {x_t.shape}")
        v = v.to(device)  # 确保 v 在同一个设备上
        for t in reversed(range(num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            with autocast(device_type='cuda'):
                epsilon_theta = model(x_t, t_batch, v)
            # print(f"epsilon_theta: {epsilon_theta.shape}")
            beta_t = betas[t]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
            sqrt_alpha_recip = 1.0 / torch.sqrt(alphas[t])
            x_t = sqrt_alpha_recip * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * epsilon_theta)
            # print(f"x_t after update: {x_t.shape}")
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta_t) * noise
                print(t)
        return x_t  # 返回生成的梅尔频谱图

videos = torch.load('videos.pt',weights_only=True)
v = videos[0].unsqueeze(0)
print(v.shape)
x_t = sample(model, v, num_timesteps, betas, device, freq_bins, time_steps)

import librosa

def mel_spectrogram_to_audio(mel_spectrogram, mean=-27.747272491455078, std=16.696313858032227, sr=16000, n_fft=1024, hop_length=512):
    # 去归一化
    mel_spectrogram = mel_spectrogram * std + mean

    positive_count = torch.sum(mel_spectrogram > 0).item()  # 正值数量
    negative_count = torch.sum(mel_spectrogram < 0).item()  # 负值数量

    print(f"正值数量: {positive_count}")
    print(f"负值数量: {negative_count}")

    # 转换为 NumPy 格式
    mel_spectrogram = mel_spectrogram.cpu().numpy()

    mel_spectrogram = np.clip(mel_spectrogram, a_min=-80, a_max=0)

    # 将对数梅尔谱转为功率谱
    mel_spectrogram = librosa.db_to_power(mel_spectrogram, ref=20)

    # 使用 Griffin-Lim 算法还原音频
    audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length)
    audio = audio.squeeze()

    # 保存为 wav 文件
    wav_filename = 'generated_audio.wav'
    sf.write(wav_filename, audio, sr)
    print(f"音频文件保存为: {wav_filename}")

    return wav_filename

mel_spectrogram_to_audio(x_t)


