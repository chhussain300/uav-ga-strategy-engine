"""Multi-objective fitness (weighted-sum scalarisation) + hard-constraint penalties.

F(x) =  w1*Coverage + w2*Detection - w3*Energy - w4*Time - w5*Hazard
        - lam1*NoFly - lam2*RangeExcess                       (all terms normalised to ~[0,1])
"""
import numpy as np
from .config import EnvConfig, UAVConfig, FitnessWeights
from .environment import Environment

HAZARD_REF = 120.0   # reference hazard-time that maps to 1.0 (calibrated on random plans)


class Evaluator:
    def __init__(self, env: Environment, uav: UAVConfig, w: FitnessWeights):
        self.env, self.uav, self.w = env, uav, w
        self.cfg: EnvConfig = env.cfg
        self.n_evals = 0

    # ------------------------------------------------------------------
    def _sample_leg_points(self, a, b, step):
        """Sample points along leg a->b; return pts and leg length."""
        d = np.linalg.norm(b - a, axis=1)
        n = np.maximum(np.ceil(d / step).astype(int), 1)
        idx = np.repeat(np.arange(len(d)), n)
        offs = np.concatenate([np.arange(k) for k in n])
        t = (offs + 0.5) / n[idx]
        pts = a[idx] + (b[idx] - a[idx]) * t[:, None]
        return pts, idx, n, d

    def evaluate(self, c, return_details=False):
        env, uav, w = self.env, self.uav, self.w
        self.n_evals += 1
        D = c.wp.shape[0]
        routes = c.routes(self.cfg.base)
        step = self.cfg.sample_step

        all_pts, all_r = [], []
        energy = np.zeros(D); tim = np.zeros(D); haz = 0.0
        nofly_hits = 0; nofly_total = 0
        for d in range(D):
            a, b = routes[d][:-1], routes[d][1:]
            pts, leg, n, dist = self._sample_leg_points(a, b, step)
            v = c.speed[d]                                   # speed multiplier per leg
            leg_time = dist / (uav.v_ref * v)
            on = c.active[d]
            pw = c.power[d] * on
            # energy: propulsion (quadratic in speed) + sensor (quadratic in power, per unit time)
            e_move = dist * (0.6 + 0.4 * v ** 2)
            e_sens = uav.sensor_cost * pw ** 2 * leg_time
            energy[d] = (e_move + e_sens).sum()
            tim[d] = leg_time.sum()
            # hazard exposure = time spent in hazardous cells (weighted by hazard intensity)
            seg_t = (leg_time / n)[leg]                       # time represented by one sample
            haz += float((env.lookup(env.hazard, pts) * seg_t).sum())
            nofly_hits += int(env.lookup(env.nofly, pts).sum()); nofly_total += len(pts)
            r = on[leg] * (uav.r_min + uav.r_gain * c.power[d][leg])
            all_pts.append(pts); all_r.append(r)

        pts = np.vstack(all_pts); rad = np.concatenate(all_r)
        live = rad > 0
        P, R = pts[live], rad[live]
        if len(P):
            # coverage: cell is covered if inside a sensor footprint (priority-weighted)
            d2 = env.cell_sq[:, None] + (P ** 2).sum(1)[None, :] - 2.0 * env.cell_xy @ P.T
            covered = (d2 <= R[None, :] ** 2).any(1).reshape(env.cfg.grid_size, -1)
            cov = float((env.priority * covered * ~env.nofly).sum() / env.total_priority)
            # detection: probability decays with distance to the sensor centre
            dt = ((env.targets[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            p = np.clip(1 - dt / (R[None, :] ** 2), 0, 1).max(1)
            det = float((p * env.target_w).sum() / env.total_target_w)
        else:
            cov = det = 0.0
            covered = np.zeros((env.cfg.grid_size,) * 2, bool)

        E_norm = float(energy.sum() / (D * uav.max_energy))
        T_norm = float(tim.max() / uav.max_time)                         # makespan
        H_norm = haz / HAZARD_REF
        nofly = nofly_hits / max(nofly_total, 1)
        excess = float((np.maximum(energy / uav.max_energy - 1, 0) + np.maximum(tim / uav.max_time - 1, 0)).sum())

        F = (w.coverage * cov + w.detection * det - w.energy * E_norm - w.time * T_norm
             - w.hazard * H_norm - w.nofly_penalty * nofly - w.range_penalty * excess)
        c.fitness = F
        c.info = dict(coverage=cov, detection=det, energy=E_norm, time=T_norm, hazard=H_norm,
                      nofly=nofly, excess=excess, feasible=(nofly == 0 and excess == 0))
        if return_details:
            c.info["covered_mask"] = covered
            c.info["energy_per_drone"] = energy; c.info["time_per_drone"] = tim
        return F
