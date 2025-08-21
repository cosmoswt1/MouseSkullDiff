import torch
from diffusers import UNet2DModel, DPMSolverMultistepScheduler # DPMSolverMultistepScheduler를 import
from torchvision.utils import save_image
import os
import math
import argparse
import numpy as np

def generate_images(args):
    """지정된 GPU에서 DPM-Solver를 사용해 이미지를 생성하고, 각 샘플에 대해 (.npy, .png) 페어를 저장"""

    # --- 1. 설정 ---
    device = f"cuda:{args.gpu_id}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 하위 폴더 분리 저장
    npy_dir = os.path.join(output_dir, "npy")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # 총 개수 및 배치 사이즈
    num_images_to_generate = args.num_images
    batch_size = args.batch_size
    # DPM-Solver는 더 적은 스텝으로도 충분합니다.
    num_inference_steps = args.num_steps

    # HU 스케일 관련 고정값
    HU_MAX = 3500.0

    # --- 2. 모델 불러오기 ---
    print(f"[GPU {args.gpu_id}] 모델을 불러오는 중...")
    unet = UNet2DModel.from_pretrained("skull_unet1ch_init")
    state = torch.load("runs/diff_512_1ch/ckpt_ep190_EMA.pth", map_location="cpu")
    unet.load_state_dict(state, strict=True)
    unet.to(device)
    unet.eval()
    print(f"[GPU {args.gpu_id}] 모델 로딩 완료.")

    # --- 4. 샘플링 루프 ---
    num_batches = math.ceil(num_images_to_generate / batch_size)
    total_generated = 0
    global_idx = args.start_index

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_images_to_generate - (i * batch_size))
        if current_batch_size <= 0:
            break

        # --- 스케줄러는 배치마다 새로 생성/리셋 (stateful 내부버퍼 방지) ---
        sched = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
        )
        sched.set_timesteps(num_inference_steps, device=device)

        print(f"[GPU {args.gpu_id}] 배치 {i+1}/{num_batches} ({current_batch_size}장) 생성 시작...")

        # 초기 잡음 ([-1,1]로 복원할 예정)
        x = torch.randn(current_batch_size, 1, 512, 512, device=device)

        # DPM-Solver 스텝 전개
        for t in sched.timesteps:
            t_cur = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v = unet(x, t_cur).sample  # v-prediction
            x = sched.step(v, t, x).prev_sample

        # 모델 출력 x: [-1,1] 범위로 클램프
        x = x.clamp(-1.0, 1.0)

        # --- 5. 배경을 -1로 강제 (스컬 마스크 추정) ---
        # 스컬(조직) 영역은 -1보다 크고, 배경은 -1 근처임을 이용해 간단 임계값 적용
        # 매우 보수적으로 -0.98 기준 사용
        with torch.no_grad():
            skull_mask = (x > -0.98)
            x = torch.where(skull_mask, x, torch.tensor(-1.0, device=device, dtype=x.dtype))

        # --- 6. 저장: (a) .npy는 HU 스케일로 복원하여 저장, (b) .png는 미리보기로 저장 ---
        # HU 복원: x \in [-1,1] -> x01 \in [0,1] -> HU \in [0, HU_MAX]
        x01 = (x + 1.0) * 0.5
        hu = (x01 * HU_MAX).float()  # [0, HU_MAX]

        # 각 샘플 개별 저장
        x_cpu = x.cpu()
        hu_cpu = hu.cpu()

        for b in range(current_batch_size):
            idx = global_idx
            global_idx += 1

            # --- (a) .npy (float32, HU 스케일) ---
            npy_path = os.path.join(npy_dir, f"skull_{idx:05d}.npy")
            np.save(npy_path, hu_cpu[b, 0].numpy().astype(np.float32))

            # --- (b) .png 프리뷰 (0..1 스케일에서 저장) ---
            # 프리뷰는 기존처럼 [-1,1] -> [0,1] 변환 결과 사용
            img01 = (x_cpu[b] + 1.0) * 0.5  # [1,512,512]
            png_path = os.path.join(png_dir, f"skull_{idx:05d}.png")
            save_image(img01, png_path)

            total_generated += 1

        print(f"[GPU {args.gpu_id}] 배치 {i+1} 저장 완료. 누적 {total_generated}장")

    print(f"✅ [GPU {args.gpu_id}] 완료! 총 {total_generated}장의 (.npy, .png) 페어를 '{output_dir}'에 저장했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU DPM-Solver Inference: export paired (.npy HU, .png preview)")
    parser.add_argument("--gpu_id", type=int, required=True, help="사용할 GPU의 ID (예: 0 또는 1)")
    parser.add_argument("--num_images", type=int, default=10000, help="이 프로세스에서 생성할 총 이미지 수 (기본: 20000)")
    parser.add_argument("--batch_size", type=int, default=8, help="한 번에 생성할 이미지 배치 크기")
    parser.add_argument("--num_steps", type=int, default=20, help="DPM-Solver inference steps (기본: 20)")
    parser.add_argument("--start_index", type=int, default=0, help="파일 인덱스 시작값 (멀티프로세스 시 충돌 방지)")
    parser.add_argument("--output_dir", type=str, default="dataset", help="(.npy, .png) 페어를 저장할 디렉토리")
    args = parser.parse_args()

    generate_images(args)