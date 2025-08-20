import torch
from diffusers import UNet2DModel

# 3채널 사전학습 모델 로드
m3 = UNet2DModel.from_pretrained("google/ddpm-celebahq-256")
sd = m3.state_dict()
cfg = m3.config

# 1채널 모델(동일 구성 재사용) 생성
m1 = UNet2DModel(
    sample_size=cfg.sample_size,
    in_channels=1,                # <- 변경
    out_channels=1,               # <- 변경
    layers_per_block=cfg.layers_per_block,
    block_out_channels=tuple(cfg.block_out_channels),
    down_block_types=tuple(cfg.down_block_types),
    up_block_types=tuple(cfg.up_block_types),
    add_attention=cfg.add_attention if hasattr(cfg, "add_attention") else None,
    # 필요 시 cfg의 다른 필드도 동일하게 전달
)

sd_new = {}

def fuse_in(w3):  # [out,3,kh,kw] -> [out,1,kh,kw]
    return w3.mean(dim=1, keepdim=True)

# 1) conv_in 처리 (입력 3 -> 1)
if "conv_in.weight" in sd and sd["conv_in.weight"].shape[1] == 3:
    sd_new["conv_in.weight"] = fuse_in(sd["conv_in.weight"])
    sd_new["conv_in.bias"]   = sd.get("conv_in.bias", torch.zeros_like(m1.state_dict()["conv_in.bias"]))
else:
    # 없거나 형태 다르면 기본값 유지
    pass

# 2) 내부 레이어: 모양 동일한건 그대로 복사
for k, v in sd.items():
    if k.startswith("conv_in") or k.startswith("conv_out"):
        continue  # 별도 처리
    if k in m1.state_dict() and m1.state_dict()[k].shape == v.shape:
        sd_new[k] = v  # 동일형상은 그대로

# 3) conv_out 처리 (출력 3 -> 1): out 채널 평균
if "conv_out.weight" in sd and sd["conv_out.weight"].shape[0] == 3:
    w3 = sd["conv_out.weight"]  # [3, C, kh, kw]
    b3 = sd.get("conv_out.bias", torch.zeros(3))
    w1 = w3.mean(dim=0, keepdim=True)  # [1, C, kh, kw]
    b1 = b3.mean().view(1)             # [1]
    sd_new["conv_out.weight"] = w1
    sd_new["conv_out.bias"]   = b1

# 4) 로드
res = m1.load_state_dict(sd_new, strict=False)
print("missing:", len(res.missing_keys), res.missing_keys[:10])
print("unexpected:", len(res.unexpected_keys), res.unexpected_keys[:10])

m1.save_pretrained("skull_unet1ch_init")
print("saved -> skull_unet1ch_init")