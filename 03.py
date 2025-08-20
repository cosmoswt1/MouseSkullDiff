import torch
from diffusers import UNet2DModel, DPMSolverMultistepScheduler # DPMSolverMultistepScheduler를 import
from torchvision.utils import save_image
import os
import math
import argparse

def generate_images(args):
    """지정된 GPU에서 DPM-Solver를 사용해 이미지를 생성하는 함수"""

    # --- 1. 설정 ---
    device = f"cuda:{args.gpu_id}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_images_to_generate = args.num_images
    batch_size = args.batch_size
    # DPM-Solver는 더 적은 스텝으로도 충분합니다.
    num_inference_steps = 20

    # --- 2. 모델 불러오기 (동일) ---
    print(f"[GPU {args.gpu_id}] 모델을 불러오는 중...")
    unet = UNet2DModel.from_pretrained("skull_unet1ch_init")
    state = torch.load("runs/diff_512_1ch/ckpt_ep190_EMA.pth", map_location="cpu")
    unet.load_state_dict(state, strict=True)
    unet.to(device)
    unet.eval()
    print(f"[GPU {args.gpu_id}] 모델 로딩 완료.")


    # --- 4. 샘플링 루프 (코드 자체는 동일) ---
    num_batches = math.ceil(num_images_to_generate / batch_size)
    total_generated = 0

    for i in range(num_batches):

        # --- 3. 스케줄러를 DPM-Solver로 변경 ---
        # 훈련 시 사용했던 파라미터를 그대로 맞춰주는 것이 중요합니다.
        sched = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
        )
        sched.set_timesteps(num_inference_steps)

        current_batch_size = min(batch_size, num_images_to_generate - (i * batch_size))
        print(f"[GPU {args.gpu_id}] 배치 {i+1}/{num_batches} ({current_batch_size}장) 생성 시작...")

        x = torch.randn(current_batch_size, 1, 512, 512, device=device)

        # sched.timesteps에 맞춰 루프를 돕니다.
        for t in sched.timesteps:
            t_cur = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # DPM-Solver는 noise 예측값을 그대로 사용합니다.
                # prediction_type="v_prediction"이므로 모델은 v를 예측합니다.
                v = unet(x, t_cur).sample
            x = sched.step(v, t, x).prev_sample

        imgs = (x.clamp(-1, 1) + 1) * 0.5
        save_path = os.path.join(output_dir, f"gpu{args.gpu_id}_batch{i+1}.png")
        save_image(imgs, save_path, nrow=5)
        total_generated += current_batch_size

    print(f"✅ [GPU {args.gpu_id}] 완료! 총 {total_generated}장의 이미지를 '{output_dir}'에 저장했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU DPM-Solver Inference")
    parser.add_argument("--gpu_id", type=int, required=True, help="사용할 GPU의 ID (예: 0 또는 1)")
    parser.add_argument("--num_images", type=int, default=50, help="이 프로세스에서 생성할 총 이미지 수")
    parser.add_argument("--batch_size", type=int, default=8, help="한 번에 생성할 이미지 배치 크기")
    parser.add_argument("--output_dir", type=str, default="samples_ep190_dpmsvr_20steps", help="이미지를 저장할 디렉토리")
    args = parser.parse_args()

    generate_images(args)