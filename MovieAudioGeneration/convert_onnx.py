import torch
from ModelZoo.ECHO41 import UNet

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
x_t = torch.randn(1, 1, 80, 1000)
t_batch = torch.full((1,), 1, dtype=torch.long)
v = torch.randn(1, 512, 768)
torch.onnx.export(model, (x_t, t_batch, v), "model.onnx")