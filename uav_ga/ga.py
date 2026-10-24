"""Genetic Algorithm engine (OOP)."""
import time
import numpy as np
from .chromosome import Chromosome
from .config import GAConfig, UAVConfig
from .fitness import Evaluator
from .operators import tournament_select, crossover, mutate, diversity


class GeneticAlgorithm:
    def __init__(self, evaluator: Evaluator, ga: GAConfig, snapshot_gens=()):
        self.ev, self.cfg = evaluator, ga
        self.env, self.uav = evaluator.env, evaluator.uav
        self.rng = np.random.default_rng(ga.seed)
        self.snapshot_gens = set(snapshot_gens)
        self.history = dict(best=[], avg=[], worst=[], diversity=[], coverage=[], mut_rate=[], evals=[])
        self.snapshots = {}
        self.best = None

    def init_population(self):
        pop = [Chromosome.random(self.env.cfg, self.uav, self.rng, self.env) for _ in range(self.cfg.pop_size)]
        for c in pop: self.ev.evaluate(c)
        return pop

    def run(self, verbose=True):
        cfg = self.cfg
        pop = self.init_population()
        n_elite = max(1, int(cfg.elite_frac * cfg.pop_size))
        n_imm = int(cfg.immigrant_frac * cfg.pop_size)
        rate = cfg.mutation_rate
        stagn, best_f = 0, -np.inf
        cooldown = 0          # generations left before the diversity safeguard may fire again
        t0 = time.time()
        for gen in range(cfg.generations + 1):
            pop.sort(key=lambda c: c.fitness, reverse=True)
            f = np.array([c.fitness for c in pop])
            div = diversity(pop, self.rng)
            self.history["best"].append(f[0]); self.history["avg"].append(f.mean())
            self.history["worst"].append(f[-1]); self.history["diversity"].append(div)
            self.history["coverage"].append(pop[0].info["coverage"]); self.history["mut_rate"].append(rate)
            self.history["evals"].append(self.ev.n_evals)
            if gen in self.snapshot_gens:
                self.snapshots[gen] = pop[0].copy(); self.ev.evaluate(self.snapshots[gen], True)
            if f[0] > best_f + 1e-6: best_f, stagn = f[0], 0
            else: stagn += 1
            if verbose and gen % 10 == 0:
                print(f"gen {gen:4d} best {f[0]:.4f} avg {f.mean():.4f} div {div:5.2f} "
                      f"cov {pop[0].info['coverage']:.3f} feas {pop[0].info['feasible']}")
            # ---- termination criteria
            if gen == cfg.generations or stagn >= cfg.stagnation_limit:
                if gen not in self.snapshot_gens: self.snapshots[gen] = pop[0].copy(); self.ev.evaluate(self.snapshots[gen], True)
                break
            # ---- adaptive mutation: increase when diversity collapses (premature convergence guard)
            low_div = div < cfg.diversity_threshold and cooldown == 0
            cooldown = max(0, cooldown - 1)
            if low_div: cooldown = 10                  # hysteresis: boost lasts 10 generations
            boost = cooldown > 0
            rate = min(0.6, cfg.mutation_rate * 2.5) if boost else cfg.mutation_rate
            # ---- next generation
            nxt = [c.copy() for c in pop[:n_elite]]
            for c, o in zip(nxt, pop[:n_elite]): c.fitness, c.info = o.fitness, o.info
            while len(nxt) < cfg.pop_size - (n_imm if low_div else 0):
                p1 = tournament_select(pop, cfg.tournament_k, self.rng)
                p2 = tournament_select(pop, cfg.tournament_k, self.rng)
                if self.rng.random() < cfg.crossover_rate:
                    c1, c2 = crossover(p1, p2, self.rng, self.uav, self.env)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    mutate(c, self.rng, rate, cfg.mutation_sigma, self.uav, self.env)
                    self.ev.evaluate(c); nxt.append(c)
            while len(nxt) < cfg.pop_size:             # random immigrants
                c = Chromosome.random(self.env.cfg, self.uav, self.rng, self.env); self.ev.evaluate(c); nxt.append(c)
            pop = nxt[:cfg.pop_size]
        pop.sort(key=lambda c: c.fitness, reverse=True)
        self.best = pop[0]; self.ev.evaluate(self.best, True)
        self.final_pop = pop
        self.runtime = time.time() - t0
        return self.best
