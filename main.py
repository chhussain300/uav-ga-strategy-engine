"""Run the full UAV GA simulation, baseline comparison and produce all figures.

    python main.py                # full run (default settings)
    python main.py --quick        # fast demo
"""
import argparse, json, os
import numpy as np
from uav_ga.config import EnvConfig, UAVConfig, FitnessWeights, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.ga import GeneticAlgorithm
from uav_ga.baseline import random_search
from uav_ga import visualize as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=100); ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--out", default="outputs")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.pop, a.gens, a.seeds = 40, 40, 2
    os.makedirs(a.out, exist_ok=True)
    env = Environment(EnvConfig()); uav, w = UAVConfig(), FitnessWeights()
    viz.plot_environment(env, f"{a.out}/00_environment.png")

    snap_gens = [0, 10, 30, 60, a.gens]
    runs, ga_curves, rs_curves, rs_results = [], [], [], []
    for s in range(a.seeds):
        ev = Evaluator(env, uav, w)
        ga = GeneticAlgorithm(ev, GAConfig(pop_size=a.pop, generations=a.gens, seed=s), snapshot_gens=snap_gens)
        print(f"\n=== GA seed {s} ==="); best = ga.run(verbose=(s == 0))
        runs.append((ga, best)); ga_curves.append((ga.history["evals"], ga.history["best"]))
        ev2 = Evaluator(env, uav, w)
        rs_best, rs_hist, _ = random_search(ev2, ev.n_evals, seed=100 + s)
        rs_results.append(rs_best); rs_curves.append(tuple(zip(*rs_hist)))
        print(f"seed {s}: GA {best.fitness:.4f} | random search {rs_best.fitness:.4f}")

    top = max(range(a.seeds), key=lambda i: runs[i][1].fitness)
    ga0, best = runs[top]
    viz.plot_convergence(ga0.history, f"{a.out}/01_fitness_convergence.png")
    snaps = {g: c for g, c in ga0.snapshots.items() if g in snap_gens or g == max(ga0.snapshots)}
    viz.plot_evolution(env, snaps, f"{a.out}/02_paths_and_coverage_evolution.png")
    viz.plot_heatmap_progress(env, snaps, f"{a.out}/03_coverage_heatmaps.png")
    viz.plot_paths(env, best, f"{a.out}/04_best_flight_paths.png", f"Best evolved plan (seed {top})")
    rs_top = max(rs_results, key=lambda c: c.fitness)
    viz.plot_vs_random(ga_curves, rs_curves, f"{a.out}/05_ga_vs_random.png")
    viz.plot_compare_paths(env, best, rs_top, f"{a.out}/06_ga_vs_random_paths.png")

    # uniqueness: mean waypoint distance between best solutions of different GA seeds
    def dist(c1, c2):  # route-order-independent: nearest-waypoint distance
        A = c1.wp.reshape(-1, 2); B = c2.wp.reshape(-1, 2)
        return float(np.mean(np.min(np.linalg.norm(A[:, None] - B[None], axis=-1), axis=1)))
    ga_d = [dist(runs[i][1], runs[j][1]) for i in range(a.seeds) for j in range(i + 1, a.seeds)]
    rs_d = [dist(rs_results[i], rs_results[j]) for i in range(a.seeds) for j in range(i + 1, a.seeds)]
    keys = ["coverage", "detection", "energy", "time", "hazard", "nofly", "excess", "feasible"]
    summ = dict(
        ga=[{"seed": i, "fitness": r[1].fitness, **{k: float(r[1].info[k]) for k in keys}, "generations": len(r[0].history["best"]) - 1, "evals": r[0].ev.n_evals} for i, r in enumerate(runs)],
        random=[{"seed": i, "fitness": c.fitness, **{k: float(c.info[k]) for k in keys}} for i, c in enumerate(rs_results)],
        uniqueness=dict(ga_between_seed_dist=float(np.mean(ga_d)), rs_between_seed_dist=float(np.mean(rs_d))))
    json.dump(summ, open(f"{a.out}/results_summary.json", "w"), indent=2)
    print("\nGA mean fitness %.3f | random %.3f" % (np.mean([r[1].fitness for r in runs]), np.mean([c.fitness for c in rs_results])))
    print("Outputs written to", a.out)


if __name__ == "__main__":
    main()
