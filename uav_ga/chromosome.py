"""Hybrid real/integer/binary chromosome.

For every drone d and leg l (K waypoints + 1 return-to-base leg):
    wp[d, k]      real  (x, y)      waypoint coordinates               (K x 2 real genes)
    speed[d, l]   real  in [v_min, v_max]  speed multiplier of leg l   (K+1 real genes)
    power[d, l]   real  in [0, 1]          sensor power on leg l       (K+1 real genes)
    active[d, l]  bit   {0, 1}             sensor ON/OFF on leg l      (K+1 binary genes)
Effective sensor radius on a leg = active * (r_min + r_gain * power).
Path of drone d: base -> wp[d,0] -> ... -> wp[d,K-1] -> base.
"""
from __future__ import annotations
import numpy as np
from .config import EnvConfig, UAVConfig


class Chromosome:
    __slots__ = ("wp", "speed", "power", "active", "fitness", "info")

    def __init__(self, wp, speed, power, active):
        self.wp, self.speed, self.power, self.active = wp, speed, power, active
        self.fitness = None
        self.info = None

    # ------------------------------------------------------------------
    @classmethod
    def random(cls, env_cfg: EnvConfig, uav: UAVConfig, rng, env=None):
        D, K, n = env_cfg.n_drones, env_cfg.n_waypoints, env_cfg.grid_size
        # Seed each drone inside its own angular sector around the base -> structured diversity
        wp = rng.uniform(0, n - 1, size=(D, K, 2))
        for d in range(D):
            lo, hi = d * n / D, (d + 1) * n / D
            wp[d, :, 0] = np.where(rng.random(K) < 0.6, rng.uniform(lo, hi, K), wp[d, :, 0])
        speed = rng.uniform(uav.v_min, uav.v_max, size=(D, K + 1))
        power = rng.uniform(0, 1, size=(D, K + 1))
        active = (rng.random((D, K + 1)) < 0.8).astype(np.int8)
        c = cls(wp, speed, power, active)
        if env is not None:
            c.wp = env.repair_point(c.wp)
        return c

    def copy(self) -> "Chromosome":
        return Chromosome(self.wp.copy(), self.speed.copy(), self.power.copy(), self.active.copy())

    # flat vector view (used for diversity measurement)
    def vector(self):
        return self.wp.ravel()

    def routes(self, base):
        """Return list of (K+2, 2) arrays: base, waypoints..., base."""
        b = np.asarray(base, float)
        return [np.vstack([b, self.wp[d], b]) for d in range(self.wp.shape[0])]
