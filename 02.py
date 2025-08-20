import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.utils import save_image
import os
import math
import argparse # 명령줄 인자 처리를 위한 라이브러리

def generate_images(args):
    """지정된 GPU에서 이미지를 생성하는 함수"""

    # --- 1. 설정 ---
    # 명령줄 인자로 받은 GPU ID로 장치 설정
    device = f"cuda:{args.gpu_id}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 이 스크립트가 생성할 이미지 수와 배치 크기
    num_images_to_generate = args.num_images
    batch_size = args.batch_size
    num_inference_steps = 50 # DDIM 스텝 수

    # --- 2. 모델 불러오기 (이전과 동일) ---
    print(f"[GPU {args.gpu_id}] 모델을 불러오는 중...")
    unet = UNet2DModel.from_pretrained("skull_unet1ch_init")
    state = torch.load("runs/diff_512_1ch/ckpt_ep190_EMA.pth", map_location="cpu")
    unet.load_state_dict(state, strict=True)
    unet.to(device)
    unet.eval()
    print(f"[GPU {args.gpu_id}] 모델 로딩 완료.")

    # --- 3. 스케줄러를 DDIM으로 설정 ---
    sched = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
    )
    sched.set_timesteps(num_inference_steps, device=device) # device 설정 추가(불필요 이동 감소 위해..)

    # --- 4. 샘플링 루프 ---
    num_batches = math.ceil(num_images_to_generate / batch_size)
    total_generated = 0

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_images_to_generate - (i * batch_size))
        print(f"[GPU {args.gpu_id}] 배치 {i+1}/{num_batches} ({current_batch_size}장) 생성 시작...")

        x = torch.randn(current_batch_size, 1, 512, 512, device=device)

        for t in sched.timesteps:
            t_cur = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v = unet(x, t_cur).sample
            x = sched.step(v, t, x).prev_sample

        imgs = (x.clamp(-1, 1) + 1) * 0.5

        # 파일명이 겹치지 않도록 GPU ID와 배치 번호를 포함하여 저장
        save_path = os.path.join(output_dir, f"gpu{args.gpu_id}_batch{i+1}.png")
        save_image(imgs, save_path, nrow=5)
        total_generated += current_batch_size

    print(f"✅ [GPU {args.gpu_id}] 완료! 총 {total_generated}장의 이미지를 '{output_dir}'에 저장했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU DDIM Inference")
    parser.add_argument("--gpu_id", type=int, required=True, help="사용할 GPU의 ID (예: 0 또는 1)")
    parser.add_argument("--num_images", type=int, default=50, help="이 프로세스에서 생성할 총 이미지 수")
    parser.add_argument("--batch_size", type=int, default=8, help="한 번에 생성할 이미지 배치 크기")
    parser.add_argument("--output_dir", type=str, default="samples_ep190_ddim_multi_gpu", help="이미지를 저장할 디렉토리")
    args = parser.parse_args()

    generate_images(args)