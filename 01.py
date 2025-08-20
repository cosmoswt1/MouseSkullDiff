import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, EMAModel
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument('--perf', action='store_true', help='Max performance mode (use more VRAM, higher throughput)')
parser.add_argument('--batch', type=int, default=None, help='Override batch size')
parser.add_argument('--grad_accum', type=int, default=None, help='Override gradient accumulation steps')
args, unknown = parser.parse_known_args()

# --- WSL2/NCCL 환경변수 ---
import os as _os
if args.perf:
    # 성능 우선: 가능하면 P2P/SHM 사용 (WSL2에서 불안정하면 다시 끄세요)
    _os.environ["NCCL_P2P_DISABLE"] = "0"
    _os.environ["NCCL_IB_DISABLE"] = "1"
    _os.environ["NCCL_SHM_DISABLE"] = "0"
    _os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
else:
    # 안정성 우선
    _os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    _os.environ.setdefault("NCCL_IB_DISABLE", "1")
    _os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    _os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
# New allocator env (old name is deprecated)
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

class SkullNPY(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "dataset_x_*.npy"), recursive=True))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = np.load(self.files[i]).astype(np.float32)
        x = torch.from_numpy(x)[None, ...]
        return x

def main(
    data_dir="data",
    out_dir="runs/diff_512_1ch",
    epochs=200, batch=10, lr=1e-4, T=1000, grad_accum=1
):
    # CLI override
    if args.batch is not None:
        batch = args.batch
    if args.grad_accum is not None:
        grad_accum = args.grad_accum

    os.makedirs(out_dir, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=args.perf)
    acc = Accelerator(gradient_accumulation_steps=grad_accum, kwargs_handlers=[ddp_kwargs])
    device = acc.device

    # 성능 최적화 옵션
    torch.backends.cudnn.benchmark = True if args.perf else False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # Decide AMP dtype (bf16/fp16) from Accelerator setting
    use_bf16 = (acc.mixed_precision == "bf16")
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if acc.mixed_precision == "fp16" else torch.float32)

    ds = SkullNPY(data_dir)
    num_workers = max(8, os.cpu_count() // 2) if args.perf else 4
    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4 if args.perf else 2,
    )

    # --- 올바른 초기화 순서 ---
    # 1. 모든 모델과 옵티마이저는 CPU에서 먼저 생성합니다.
    base_unet = UNet2DModel.from_pretrained("skull_unet1ch_init")
    opt = torch.optim.AdamW(base_unet.parameters(), lr=lr, weight_decay=1e-4)

    # 2. acc.prepare()가 unet과 opt를 각자 알맞은 GPU로 보냅니다.
    unet, opt, dl = acc.prepare(base_unet, opt, dl)
    unwrap = acc.unwrap_model(unet)

    # 3. prepare가 끝난 후, 언랩된 원본 파라미터로 EMA 생성 (이 시점에 모델은 GPU에 있음)
    ema = EMAModel(unwrap.parameters(), decay=0.9999)

    # 메모리/성능 전략
    # 항상 gradient checkpointing 활성화: 대규모 512x512 UNet에서 배치/피크메모리 절감 효과가 큼
    if hasattr(unwrap, "enable_gradient_checkpointing"):
        unwrap.enable_gradient_checkpointing()

    if args.perf:
        # 성능 우선: attention slicing은 끄고, 채널라스트 입력만 사용
        if hasattr(unwrap, "disable_attention_slicing"):
            try:
                unwrap.disable_attention_slicing()
            except Exception:
                pass
    else:
        # 메모리 우선: attention slicing 켬
        if hasattr(unwrap, "enable_attention_slicing"):
            try:
                unwrap.enable_attention_slicing("auto")
            except TypeError:
                unwrap.enable_attention_slicing()

    noise_sched = DDPMScheduler(
        num_train_timesteps=T,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )
    alphas_cumprod = noise_sched.alphas_cumprod.to(device)

    global_step = 0
    for ep in range(1, epochs+1):
        unet.train()
        pbar = tqdm(dl, disable=not acc.is_local_main_process)
        for x0 in pbar:
            with acc.accumulate(unet):
                b = x0.size(0)
                t = torch.randint(0, noise_sched.config.num_train_timesteps, (b,), device=device, dtype=torch.long)

                # Move/cast inputs to AMP dtype first to avoid holding both fp32 & fp16/bf16 copies
                x0 = x0.to(device=device, dtype=amp_dtype, non_blocking=True)
                if args.perf:
                    x0 = x0.to(memory_format=torch.channels_last)
                noise = torch.randn_like(x0, dtype=amp_dtype, device=device)
                xt = noise_sched.add_noise(x0, noise, t)
                if args.perf:
                    xt = xt.to(memory_format=torch.channels_last)

                pred = unet(xt, t).sample
                # v-pred target: v = alpha * eps - sigma * x0, where alpha = sqrt(a_bar), sigma = sqrt(1 - a_bar)
                a_bar_t = alphas_cumprod.gather(0, t)
                alpha_t = a_bar_t.sqrt().to(dtype=amp_dtype)
                sigma_t = (1.0 - a_bar_t).sqrt().to(dtype=amp_dtype)
                v_target = alpha_t[:, None, None, None] * noise - sigma_t[:, None, None, None] * x0
                loss = (pred - v_target).pow(2).mean()

                acc.backward(loss)
                opt.step(); opt.zero_grad()

                if acc.sync_gradients:
                    ema.step(unwrap.parameters())
                    global_step += 1
                    del xt, pred
                if acc.is_local_main_process:
                    pbar.set_description(f"ep{ep} loss {loss.item():.4f}")

        if acc.is_local_main_process and (ep % 5 == 0 or ep == epochs):
            with torch.no_grad():
                torch.save(unwrap.state_dict(), os.path.join(out_dir, f"ckpt_ep{ep:03d}_ONLINE.pth"))

                ema.store(unwrap.parameters())
                ema.copy_to(unwrap.parameters())

                # <<< 수정: DDP 래퍼(unet)가 아닌 실제 모델(unwrap)을 eval 모드로 변경해야 합니다.
                unwrap.eval()
                x = torch.randn(4, 1, 512, 512, device=device, dtype=amp_dtype)
                if args.perf:
                    x = x.to(memory_format=torch.channels_last)
                for ti in reversed(range(noise_sched.config.num_train_timesteps)):
                    tcur = torch.full((4,), ti, device=device, dtype=torch.long)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        # <<< 수정: EMA 가중치가 적용된 실제 모델(unwrap)로 샘플링을 진행해야 합니다.
                        v = unwrap(x, tcur).sample
                    x = noise_sched.step(v, ti, x).prev_sample
                img = ((x.clamp(-1, 1) + 1) * 0.5)
                save_image(img, os.path.join(out_dir, f"sample_ep{ep:03d}.png"), nrow=4)
                
                torch.save(unwrap.state_dict(), os.path.join(out_dir, f"ckpt_ep{ep:03d}_EMA.pth"))
                ema.restore(unwrap.parameters())

if __name__ == "__main__":
    main()