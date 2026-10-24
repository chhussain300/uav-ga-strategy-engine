"""Hyper-parameter sensitivity analysis (population size, mutation rate, selection pressure)."""
import argparse, json, os
import numpy as np
from uav_ga.config import EnvConfig, UAVConfig, FitnessWeights, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.ga import GeneticAlgorithm
from uav_ga import visualize as viz

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--gens", type=int, default=100)
ap.add_argument("--out", default="outputs"); a = ap.parse_args()
env = Environment(EnvConfig()); uav, w = UAVConfig(), FitnessWeights()
grid = {"Population size": ("pop_size", [20, 50, 100, 200]),
        "Mutation rate": ("mutation_rate", [0.02, 0.10, 0.20, 0.40]),
        "Tournament size k (selection pressure)": ("tournament_k", [2, 3, 5, 10])}
res, raw = {}, {}
for name, (key, vals) in grid.items():
    res[name] = {}
    for v in vals:
        fs, evs = [], []
        for s in range(a.seeds):
            kw = dict(generations=a.gens, seed=s, stagnation_limit=10**9); kw[key] = v
            if key != "pop_size": kw["pop_size"] = 60
            ev = Evaluator(env, uav, w); g = GeneticAlgorithm(ev, GAConfig(**kw)); g.run(verbose=False)
            fs.append(g.best.fitness); evs.append(ev.n_evals)
        res[name][str(v)] = fs
        raw[f"{name}={v}"] = dict(mean=float(np.mean(fs)), std=float(np.std(fs)), evals=float(np.mean(evs)))
        print(f"{name:42s} {v:>6}: {np.mean(fs):.3f} +- {np.std(fs):.3f}  (evals {np.mean(evs):.0f})")
viz.plot_sensitivity(res, f"{a.out}/07_hyperparameter_sensitivity.png")
json.dump(raw, open(f"{a.out}/sensitivity_results.json", "w"), indent=2)
