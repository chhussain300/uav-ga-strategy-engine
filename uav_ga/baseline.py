"""Baseline: pure random search with the same evaluation budget as the GA."""
import numpy as np
from .chromosome import Chromosome


def random_search(evaluator, budget, seed=0, log_every=100):
    rng = np.random.default_rng(seed)
    best, hist, allf = None, [], []
    for i in range(budget):
        c = Chromosome.random(evaluator.env.cfg, evaluator.uav, rng, evaluator.env)
        evaluator.evaluate(c); allf.append(c.fitness)
        if best is None or c.fitness > best.fitness: best = c
        if i % log_every == 0 or i == budget - 1: hist.append((i, best.fitness))
    evaluator.evaluate(best, True)
    return best, hist, np.array(allf)
