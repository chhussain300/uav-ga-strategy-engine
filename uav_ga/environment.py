"""Discretised hazard grid: priority zones, hazard pockets, no-fly zones, targets."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import EnvConfig


class Environment:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        n = cfg.grid_size
        rng = np.random.default_rng(cfg.env_seed)
        yy, xx = np.mgrid[0:n, 0:n]
        self.cell_xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)  # (n*n, 2) (x, y)

        self.cell_sq = (self.cell_xy ** 2).sum(1)

        # --- priority map: disaster zones (gaussian blobs, values 0..1) ---------------------
        self.priority = np.zeros((n, n))
        self.zone_centres = [(12, 38, 5.5, 1.0), (38, 36, 6.5, 0.9), (30, 12, 5.0, 0.8),
                             (44, 8, 3.5, 0.6), (20, 22, 4.0, 0.5)]
        for cx, cy, s, w in self.zone_centres:
            self.priority += w * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
        self.priority = np.clip(self.priority, 0, 1)
        self.priority[self.priority < 0.05] = 0.05          # whole area has small baseline value

        # --- hazard pockets (high wind / radiation), values 0..1 ---------------------------
        self.hazard = np.zeros((n, n))
        for cx, cy, s, w in [(24, 30, 5.0, 1.0), (10, 15, 4.0, 0.8), (40, 22, 4.5, 0.9)]:
            self.hazard += w * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
        self.hazard = np.clip(self.hazard, 0, 1)

        # --- no-fly zones (blocked cells = 1) ------------------------------------------------
        self.nofly = np.zeros((n, n), dtype=bool)
        for x0, x1, y0, y1 in [(18, 24, 8, 14), (28, 33, 26, 32), (6, 11, 26, 31), (40, 46, 38, 44)]:
            self.nofly[y0:y1, x0:x1] = True
        bx, by = map(int, cfg.base)
        self.nofly[max(0, by - 3):by + 4, max(0, bx - 3):bx + 4] = False  # keep base clear

        # nearest free cell lookup (used by the repair operator)
        _, (iy, ix) = distance_transform_edt(self.nofly, return_indices=True)
        self.nearest_free = np.stack([ix, iy], axis=-1).astype(float)     # [y, x] -> (x, y)

        # --- hidden targets, sampled proportional to priority outside no-fly zones ---------------
        p = self.priority.copy(); p[self.nofly] = 0; p = (p ** 2).ravel(); p /= p.sum()
        idx = rng.choice(n * n, size=cfg.n_targets, replace=False, p=p)
        self.targets = self.cell_xy[idx]
        self.target_w = self.priority.ravel()[idx]

        self.total_priority = float((self.priority * ~self.nofly).sum())
        self.total_target_w = float(self.target_w.sum())

    # ------------------------------------------------------------------------------------
    def lookup(self, grid, pts):
        """Nearest-cell lookup of a (n,n) grid at float points (x, y)."""
        n = self.cfg.grid_size
        ix = np.clip(np.rint(pts[:, 0]).astype(int), 0, n - 1)
        iy = np.clip(np.rint(pts[:, 1]).astype(int), 0, n - 1)
        return grid[iy, ix]

    def repair_point(self, pts):
        """Clip to the map and push points inside no-fly zones to the nearest free cell."""
        n = self.cfg.grid_size
        pts = np.clip(pts, 0, n - 1)
        shp = pts.shape
        flat = pts.reshape(-1, 2)
        blocked = self.lookup(self.nofly, flat)
        if blocked.any():
            ix = np.rint(flat[blocked, 0]).astype(int); iy = np.rint(flat[blocked, 1]).astype(int)
            flat[blocked] = self.nearest_free[iy, ix]
        return flat.reshape(shp)
