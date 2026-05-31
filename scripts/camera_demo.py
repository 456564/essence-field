"""实时摄像头 — 持久粒子 + 2×2 仪表盘"""
import cv2, torch, numpy as np, time, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.operators import PhysicalOperatorLayer
from src.essence_space import EssenceSpace
from src.simulation import ParticleSimulator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 128
SPAWN_RATE = 15        # 新增粒子数/模拟帧
MAX_PARTICLES = 500    # 粒子上限
FRAME_SKIP = 2


def to_heatmap(field, size=None, cmap=cv2.COLORMAP_JET):
    if size is not None: field = cv2.resize(field, size)
    f = (np.clip(field, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(f, cmap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--video', type=str)
    args = parser.parse_args()

    ops = PhysicalOperatorLayer().to(DEVICE)
    sim = ParticleSimulator()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if not args.video else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Camera {args.cam} failed. Trying 0..4:")
        for i in range(5):
            t = cv2.VideoCapture(i)
            if t.isOpened(): print(f"  {i}: OK"); t.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Persistent particle state (GPU tensors)
    px, py, pa = torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE), torch.empty(0, dtype=torch.bool, device=DEVICE)

    last_ju = last_vp = last_wall = None
    last_retention = 0.0
    frame_count = 0; fps = 0; t0 = time.time()

    print("Camera running. r=reset particles, q=quit, s=screenshot")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        h, w = frame.shape[:2]; s = IMG_SIZE / max(h, w)
        frame_s = cv2.resize(frame, (int(w*s), int(h*s)))
        img_rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = img_rgb.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):   # Reset particles
            px, py, pa = torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE), torch.empty(0, dtype=torch.bool, device=DEVICE)
            print("Reset particles")

        if frame_count % FRAME_SKIP == 0:
            x = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(DEVICE)
            space = EssenceSpace.from_image(x, ops)

            # Step existing particles
            if pa.any():
                px, py, pa = sim.step(px, py, pa, space)

            # Spawn new particles (up to MAX)
            current = pa.sum().item()
            to_spawn = min(SPAWN_RATE, MAX_PARTICLES - current)
            if to_spawn > 0:
                sx, sy, sa = sim.spawn(space, to_spawn)
                px = torch.cat([px, sx]) if len(px) > 0 else sx
                py = torch.cat([py, sy]) if len(py) > 0 else sy
                pa = torch.cat([pa, sa]) if len(pa) > 0 else sa

            # Trim excess
            if pa.sum().item() > MAX_PARTICLES:
                keep = min(MAX_PARTICLES, len(px))
                px, py, pa = px[:keep], py[:keep], pa[:keep]

            retention, trapped = sim.compute_retention(px, py, pa, space)
            last_retention = retention

            last_ju = space.get('ju')[0,0].cpu().numpy()
            last_vp = space.get('void_prob')[0,0].cpu().numpy()
            last_wall = space.wall_mask[0,0].cpu().numpy()

        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - t0 + 1e-6); t0 = time.time()

        # ── Dashboard ──
        ds_W, ds_H = W*2, H*2

        # P1: Original + wall
        p1 = cv2.resize(frame_s, (ds_W, ds_H))
        if last_wall is not None:
            w=(last_wall*255).astype(np.uint8); w=cv2.resize(w,(ds_W,ds_H))
            p1=cv2.addWeighted(p1,0.7,cv2.applyColorMap(w,cv2.COLORMAP_HOT),0.3,0)
        cv2.putText(p1,"Original+Wall",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # P2: Ju
        p2=to_heatmap(last_ju,(ds_W,ds_H)) if last_ju is not None else np.zeros((ds_H,ds_W,3),np.uint8)
        cv2.putText(p2,f"Ju mean={last_ju.mean():.2f}" if last_ju is not None else "Ju",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # P3: VoidProb
        p3=to_heatmap(last_vp,(ds_W,ds_H),cv2.COLORMAP_HOT) if last_vp is not None else np.zeros((ds_H,ds_W,3),np.uint8)
        cv2.putText(p3,f"VoidProb mean={last_vp.mean():.2f}" if last_vp is not None else "Void",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # P4: Particles + density
        p4=cv2.resize(frame_s,(ds_W,ds_H))
        if pa.any():
            xn=px.cpu().numpy(); yn=py.cpu().numpy(); an=pa.cpu().numpy()
            tr=trapped.cpu().numpy() if 'trapped' in dir() else np.zeros(len(px))
            density=np.zeros((ds_H,ds_W),np.float32)
            sx,sy=ds_W/W,ds_H/H
            for i in range(len(xn)):
                xi,yi=int(xn[i]*sx),int(yn[i]*sy)
                if 0<=yi<ds_H and 0<=xi<ds_W:
                    if tr[i]:
                        cv2.circle(p4,(xi,yi),4,(0,255,0),-1); density[yi,xi]+=2
                    elif an[i]:
                        cv2.circle(p4,(xi,yi),2,(0,255,255),1); density[yi,xi]+=0.3
            if density.max()>0:
                density=cv2.GaussianBlur(density,(21,21),7)
                density/=max(density.max(),1)
                dh=to_heatmap(density,cmap=cv2.COLORMAP_HOT)
                p4=cv2.addWeighted(p4,0.6,dh,0.4,0)
        cv2.putText(p4,f"Particles: {pa.sum().item()}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # Bar
        color=(0,255,0) if last_retention>0.5 else (0,165,255)
        label="CONTAINER" if last_retention>0.5 else "OPEN"
        bw=int(ds_W*2*min(last_retention,1.0))
        bar=np.zeros((30,ds_W*2,3),np.uint8); bar[:,:bw]=color
        cv2.putText(bar,f"{label} ret={last_retention:.2f} FPS={fps:.0f} [r=reset]",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        row1=np.hstack([p1,p2]); row2=np.hstack([p3,p4])
        cv2.imshow('Particle Simulation',np.vstack([row1,row2,bar]))

        if key == ord('s'):
            out=Path(__file__).resolve().parent.parent/'test_output'/f'camera_{time.strftime("%H%M%S")}.png'
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out),np.vstack([row1,row2,bar]))
            print(f"Saved: {out}")

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
