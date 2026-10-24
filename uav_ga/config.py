"""Central configuration (all tunable constants live here)."""
from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_size: int = 50          # 50 x 50 grid of cells
    n_drones: int = 3            # fleet size
    n_waypoints: int = 10        # waypoints per drone (excluding return-to-base leg)
    env_seed: int = 7            # fixed seed => same disaster map every run
    n_targets: int = 30          # hidden targets (survivors / hot-spots) to detect
    sample_step: float = 2.0     # spacing (cells) used to sample along each flight leg
    base: tuple = (2.0, 2.0)     # launch / recovery point of every UAV


@dataclass
class UAVConfig:
    max_energy: float = 260.0    # hard range limit (energy units) per drone
    max_time: float = 140.0      # hard flight-time limit per drone (time units)
    v_min: float = 0.6           # relative speed multiplier bounds
    v_max: float = 1.6
    v_ref: float = 1.0           # cells per time-unit at multiplier 1.0
    r_min: float = 2.0           # sensor footprint radius at power 0 (when active)
    r_gain: float = 4.5          # extra radius gained at full sensor power
    sensor_cost: float = 1.2     # energy per time-unit at full sensor power (quadratic)


@dataclass
class FitnessWeights:
    coverage: float = 1.00       # w1  priority-weighted coverage
    detection: float = 0.70      # w2  target detection rate
    energy: float = 0.30         # w3  total energy
    time: float = 0.20           # w4  mission makespan
    hazard: float = 0.45         # w5  exposure to hazard pockets
    nofly_penalty: float = 6.0   # lambda1  fraction of path samples inside no-fly zone
    range_penalty: float = 3.0   # lambda2  normalised energy/time excess over hard limits


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 150
    tournament_k: int = 5        # selection pressure
    crossover_rate: float = 0.85
    mutation_rate: float = 0.05  # per-gene-group probability
    mutation_sigma: float = 4.0  # std-dev (cells) of gaussian waypoint perturbation
    elite_frac: float = 0.04
    immigrant_frac: float = 0.08 # random immigrants injected when diversity collapses
    diversity_threshold: float = 4.0
    stagnation_limit: int = 60   # termination: no improvement for this many generations
    seed: int = 0
