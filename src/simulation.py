"""扰动推演引擎 — 粒子洒落 + 滞留率计算"""
import torch
import torch.nn.functional as F
import numpy as np


class ParticleSimulator:
    """
    粒子洒落模拟：重力驱动 + 遇墙反弹/crawl。

    粒子从图像顶部随机投放，根据墙掩码演化，
    最终根据 ju 场计算滞留率。
    """

    def __init__(self, num_particles=500, max_steps=2000):
        self.num_particles = num_particles
        self.max_steps = max_steps

    def simulate(self, essence_space):
        """
        粒子在高 ju 区初始化，受重力下落，看能否逃出围合区。
        返回: (final_x, final_y, active, frames)
        """
        from .essence_space import EssenceSpace as ES
        wall = essence_space.wall_mask  # [1, 1, H, W]
        ju = essence_space.get('ju')    # [1, 1, H, W]
        H, W = wall.shape[-2:]
        device = wall.device

        # Spawn in high-ju regions (potential cavities)
        ju_np = ju[0, 0].cpu().numpy()
        candidates = np.argwhere(ju_np > 0.3)  # (y, x) pairs
        if len(candidates) < self.num_particles:
            # Not enough ju pixels, pad with random positions
            rand_y = np.random.randint(0, H, self.num_particles)
            rand_x = np.random.randint(0, W, self.num_particles)
            n_ju = min(len(candidates), self.num_particles)
            chosen = np.random.choice(len(candidates), n_ju, replace=False)
            rand_y[:n_ju] = candidates[chosen, 0]
            rand_x[:n_ju] = candidates[chosen, 1]
        else:
            chosen = np.random.choice(len(candidates), self.num_particles, replace=True)
            rand_y = candidates[chosen, 0]
            rand_x = candidates[chosen, 1]

        x = torch.from_numpy(rand_x.astype(np.float32)).to(device)
        y = torch.from_numpy(rand_y.astype(np.float32)).to(device)
        active = torch.ones(self.num_particles, dtype=torch.bool, device=device)

        # 可选：保存中间帧
        frames = []

        # Random velocity per particle for random walk
        vx = torch.zeros(self.num_particles, device=device)
        vy = torch.zeros(self.num_particles, device=device)

        for step in range(self.max_steps):
            if not active.any():
                break

            # Random walk with gravity bias
            # Each active particle picks a random direction with gravity down
            n_active = active.sum().item()
            if n_active == 0:
                break

            # Generate random moves: -1, 0, +1 per axis
            moves_y = torch.randint(-1, 2, (self.num_particles,), device=device).float()
            moves_x = torch.randint(-1, 2, (self.num_particles,), device=device).float()
            # Gravity bias: 80% chance to go down, 20% random
            grav_mask = torch.rand(self.num_particles, device=device) < 0.8
            moves_y = torch.where(grav_mask, torch.ones_like(moves_y), moves_y)

            new_x = (x + moves_x).clamp(0, W-1).long()
            new_y = (y + moves_y).clamp(0, H-1).long()

            # Check if new position is blocked by wall
            blocked = wall[0, 0, new_y, new_x] > 0.5

            # Move if not blocked
            x = torch.where(~blocked & active, new_x.float(), x)
            y = torch.where(~blocked & active, new_y.float(), y)

            # If blocked, try horizontal only (crawl along wall)
            h_new_x = (x + moves_x.sign()).clamp(0, W-1).long()
            h_blocked = wall[0, 0, y.long().clamp(0, H-1), h_new_x] > 0.5
            x = torch.where(blocked & ~h_blocked & active, h_new_x.float(), x)

            # Mark stuck particles (blocked in all directions for many steps)
            # Just mark as inactive if blocked horizontally too
            stuck = blocked & h_blocked & active
            active = active & ~stuck

            # Remove out-of-bounds
            out = (y >= H-1) | (y <= 0) | (x <= 0) | (x >= W-1)
            active = active & ~out

            if step % 200 == 0:
                frames.append((x.clone(), y.clone(), active.clone()))

        # 最终帧
        frames.append((x.clone(), y.clone(), active.clone()))
        return x, y, active, frames

    def compute_retention(self, x, y, active, essence_space, ju_thresh=0.3):
        """滞留率：粒子停止后仍在高 ju 区的比例"""
        ju_field = essence_space.get('ju')
        H, W = ju_field.shape[-2], ju_field.shape[-1]
        in_bounds = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        x_l = x.long().clamp(0, W-1)
        y_l = y.long().clamp(0, H-1)
        ju_vals = ju_field[0, 0, y_l, x_l]
        stopped = ~active
        in_ju = ju_vals > ju_thresh
        trapped = stopped & in_ju & in_bounds
        if (~active).any():
            return trapped.float().mean().item(), trapped
        return 0.0, trapped
