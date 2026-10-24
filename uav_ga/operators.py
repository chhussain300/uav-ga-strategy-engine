"""Selection, crossover, mutation (spatially aware) and diversity tools."""
import numpy as np
from .chromosome import Chromosome


# ----------------------------------------------------------------- selection
def tournament_select(pop, k, rng):
    idx = rng.integers(0, len(pop), size=k)
    return pop[max(idx, key=lambda i: pop[i].fitness)]


# ----------------------------------------------------------------- crossover
def crossover(p1: Chromosome, p2: Chromosome, rng, uav, env):
    """Hybrid crossover applied drone-by-drone, choosing one of three spatial operators:
       1. segment swap     - exchange a contiguous run of waypoints (and their legs) between parents
       2. geometric blend  - child waypoint = alpha*p1 + (1-alpha)*p2 (arithmetic/geometric crossover)
       3. whole-route swap - child inherits the complete route of one parent for that drone
    Binary sensor genes use uniform crossover.
    """
    c1, c2 = p1.copy(), p2.copy()
    D, K = p1.wp.shape[:2]
    for d in range(D):
        op = rng.integers(0, 3)
        if op == 0:                                   # segment swap
            a, b = sorted(rng.integers(0, K + 1, size=2))
            if a == b: b = min(K, a + 1)
            for arr in ("wp",):
                t = getattr(c1, arr)[d, a:b].copy()
                getattr(c1, arr)[d, a:b] = getattr(c2, arr)[d, a:b]
                getattr(c2, arr)[d, a:b] = t
            for arr in ("speed", "power", "active"):   # legs a..b
                t = getattr(c1, arr)[d, a:b + 1].copy()
                getattr(c1, arr)[d, a:b + 1] = getattr(c2, arr)[d, a:b + 1]
                getattr(c2, arr)[d, a:b + 1] = t
        elif op == 1:                                 # geometric blend (per-waypoint alpha)
            al = rng.random((K, 1))
            w1 = al * p1.wp[d] + (1 - al) * p2.wp[d]
            w2 = (1 - al) * p1.wp[d] + al * p2.wp[d]
            c1.wp[d], c2.wp[d] = w1, w2
            be = rng.random()
            c1.speed[d] = be * p1.speed[d] + (1 - be) * p2.speed[d]
            c2.speed[d] = (1 - be) * p1.speed[d] + be * p2.speed[d]
            c1.power[d] = be * p1.power[d] + (1 - be) * p2.power[d]
            c2.power[d] = (1 - be) * p1.power[d] + be * p2.power[d]
        else:                                         # whole-route swap
            for arr in ("wp", "speed", "power", "active"):
                t = getattr(c1, arr)[d].copy()
                getattr(c1, arr)[d] = getattr(c2, arr)[d]
                getattr(c2, arr)[d] = t
        mask = rng.random(K + 1) < 0.5                # uniform crossover for the bit-string
        t = c1.active[d, mask].copy(); c1.active[d, mask] = c2.active[d, mask]; c2.active[d, mask] = t
    for c in (c1, c2):
        c.wp = env.repair_point(c.wp)
        c.fitness = c.info = None
    return c1, c2


# ------------------------------------------------------------------ mutation
def mutate(c: Chromosome, rng, rate, sigma, uav, env):
    """Spatial mutation operators (each fires independently with probability `rate`):
       gaussian shift - perturb a waypoint by N(0, sigma^2)   (local exploitation)
       random reset   - teleport a waypoint to a uniformly random cell (global exploration)
       swap           - swap two waypoints of a route (re-orders the tour)
       reverse        - reverse a sub-tour (2-opt style move that removes crossings)
       speed/power    - gaussian tweak
       bit flip       - toggle sensor ON/OFF
    """
    D, K = c.wp.shape[:2]
    n = env.cfg.grid_size
    for d in range(D):
        for k in range(K):
            if rng.random() < rate:
                c.wp[d, k] += rng.normal(0, sigma, 2)
            if rng.random() < rate * 0.15:
                c.wp[d, k] = rng.uniform(0, n - 1, 2)
        if rng.random() < rate:
            i, j = rng.integers(0, K, 2); c.wp[d, [i, j]] = c.wp[d, [j, i]]
        if rng.random() < rate:
            i, j = sorted(rng.integers(0, K, 2)); c.wp[d, i:j + 1] = c.wp[d, i:j + 1][::-1].copy()
        m = rng.random(K + 1) < rate
        c.speed[d, m] += rng.normal(0, 0.2, m.sum())
        m = rng.random(K + 1) < rate
        c.power[d, m] += rng.normal(0, 0.2, m.sum())
        m = rng.random(K + 1) < rate * 0.5
        c.active[d, m] ^= 1
    c.speed = np.clip(c.speed, uav.v_min, uav.v_max)
    c.power = np.clip(c.power, 0, 1)
    c.wp = env.repair_point(c.wp)                     # repair mechanism (no-fly + map bounds)
    c.fitness = c.info = None
    return c


# ----------------------------------------------------------------- diversity
def diversity(pop, rng, n_pairs=200):
    """Mean waypoint distance between random pairs of individuals (cells)."""
    i = rng.integers(0, len(pop), n_pairs); j = rng.integers(0, len(pop), n_pairs)
    return float(np.mean([np.linalg.norm(pop[a].wp - pop[b].wp, axis=-1).mean() for a, b in zip(i, j)]))
