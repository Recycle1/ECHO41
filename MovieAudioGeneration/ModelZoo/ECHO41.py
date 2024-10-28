import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb  # 输出形状为 [Batch Size, dim * 4]

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.context_dim = context_dim or dim

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(self.context_dim, dim)
        self.value = nn.Linear(self.context_dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, N, C = x.shape  # x: [Batch, Seq_len, Dim]
        q = self.query(x)  # [B, N, C]
        k = self.key(context)  # [B, S, C]
        v = self.value(context)  # [B, S, C]

        # 多头注意力
        q = q.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, heads, N, dim_head]
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, heads, S, dim_head]
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, heads, S, dim_head]

        # 计算注意力得分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)  # [B, heads, N, S]
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, v)  # [B, heads, N, dim_head]
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        out = self.out(attn_output)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False, num_heads=8):
        super(ResBlock, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = CrossAttention(out_channels, context_dim=768, num_heads=num_heads)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t, context=None):
        h = self.conv1(x)
        h = self.norm1(h)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)

        if self.use_attention and context is not None:
            B, C, H, W = h.shape
            h_flat = h.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
            h_attn = self.attention(h_flat, context)  # [B, H*W, C]
            h = h_attn.transpose(1, 2).view(B, C, H, W)

        h = h + self.shortcut(x)
        h = F.silu(h)
        return h

class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels_final=1,
                 base_channels=64,
                 channel_mults=(1, 2, 4, 8, 8),
                 num_res_blocks=2,
                 time_emb_dim=256,
                 num_heads=8):
        super(UNet, self).__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim // 4)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_channels = []

        channels = base_channels

        # 编码器
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResBlock(channels, out_channels, time_emb_dim, use_attention=False)
                )
                self.skip_channels.append(out_channels)
                channels = out_channels
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample(channels))

        # 瓶颈层
        self.mid = ResBlock(channels, channels, time_emb_dim, use_attention=True, num_heads=num_heads)

        # 解码器
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            self.ups.append(Upsample(channels))
            for _ in range(num_res_blocks):
                skip_channels = self.skip_channels.pop()
                self.ups.append(
                    ResBlock(channels + skip_channels, out_channels, time_emb_dim, use_attention=False)
                )
                channels = out_channels

        # 输出卷积层
        print(out_channels_final)
        self.output_conv = nn.Conv2d(channels, out_channels_final, kernel_size=3, padding=1)

    def forward(self, x, t, context):
        t_emb = self.time_embedding(t)
        x = self.input_conv(x)

        skips = []

        # 编码器路径
        for module in self.downs:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
                skips.append(x)
            else:
                x = module(x)

        # 瓶颈层
        x = self.mid(x, t_emb, context)

        # 解码器路径
        for module in self.ups:
            if isinstance(module, Upsample):
                x = module(x)
            else:
                skip = skips.pop()
                # 调整 x 的尺寸以匹配 skip
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
                x = torch.cat([x, skip], dim=1)
                x = module(x, t_emb)

        x = self.output_conv(x)
        # print(x.shape)
        return x
