"""实时摄像头粒子推演 — 容器功能检测"""
import cv2, torch, numpy as np, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.operators import PhysicalOperatorLayer
from src.essence_space import EssenceSpace
from src.simulation import ParticleSimulator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 256        # 处理分辨率
N_PARTICLES = 150     # 粒子数
FRAME_SKIP = 3        # 每 N 帧跑一次模拟

def main():
    ops = PhysicalOperatorLayer().to(DEVICE)
    sim = ParticleSimulator(num_particles=N_PARTICLES, max_steps=200)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Try: python scripts/camera_demo.py --video test.mp4")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_particles = None
    last_retention = 0.0
    frame_count = 0
    fps = 0; t0 = time.time()

    print("Camera running. Press 'q' to quit, 's' to screenshot.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Resize + RGB
        h, w = frame.shape[:2]; s = IMG_SIZE / max(h, w)
        frame_s = cv2.resize(frame, (int(w*s), int(h*s)))
        img_rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = img_rgb.shape[:2]

        if frame_count % FRAME_SKIP == 0:
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            space = EssenceSpace.from_image(x, ops)
            fx, fy, active, _ = sim.simulate(space)
            retention, trapped = sim.compute_retention(fx, fy, active, space)
            last_retention = retention
            fx_np = fx.cpu().numpy(); fy_np = fy.cpu().numpy()
            tr_np = trapped.cpu().numpy() if isinstance(trapped, torch.Tensor) else trapped
            ac_np = active.cpu().numpy()
            last_particles = (fx_np, fy_np, ac_np, tr_np)

        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - t0 + 1e-6); t0 = time.time()

        # Draw
        display = frame_s.copy()
        display = cv2.resize(display, (W, H))

        if last_particles is not None:
            fx_np, fy_np, ac_np, tr_np = last_particles
            for i in range(len(fx_np)):
                xi, yi = int(fx_np[i]), int(fy_np[i])
                if yi >= H or xi >= W or yi < 0 or xi < 0: continue
                if tr_np[i]:
                    cv2.circle(display, (xi, yi), 3, (0, 255, 0), -1)
                elif ac_np[i]:
                    cv2.circle(display, (xi, yi), 2, (255, 255, 0), -1)

        # Overlay text
        color = (0, 255, 0) if last_retention > 0.5 else (0, 165, 255)
        label = "CONTAINER" if last_retention > 0.5 else "OPEN"
        cv2.putText(display, f"{label} (ret={last_retention:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(display, f"FPS: {fps:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        display_big = cv2.resize(display, (W*2, H*2))
        cv2.imshow('Particle Simulation', display_big)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'):
            out = Path(__file__).resolve().parent.parent / 'test_output' / f'camera_{time.strftime("%H%M%S")}.png'
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), display_big)
            print(f"Saved: {out}")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
