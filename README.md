# UAV Strategy Engine – Multi-Objective Genetic Algorithm

Plans deployment positions, sensor schedules and waypoint trajectories for a fleet of 3 UAVs over a
50×50 hazard grid (priority disaster zones, hazard pockets, no-fly zones, hidden targets).
Course: Artificial Intelligence, BSCS-5, Bahria University Lahore – Assignment 2 (CLO4).

## Structure
```
uav_ga/config.py       all constants (grid, UAV limits, fitness weights, GA hyper-parameters)
uav_ga/environment.py  grid: priority map, hazard map, no-fly mask, targets, repair lookup
uav_ga/chromosome.py   hybrid chromosome (real waypoints + real speed/power + binary sensor ON/OFF)
uav_ga/fitness.py      multi-objective fitness + penalty-based hard constraints
uav_ga/operators.py    tournament selection, 3 spatial crossovers, 6 mutations, diversity metric
uav_ga/ga.py           GA engine: elitism, adaptive mutation, random immigrants, termination
uav_ga/baseline.py     random-search baseline (same evaluation budget)
uav_ga/visualize.py    all plots
main.py                full run + baseline comparison + figures
experiments.py         hyper-parameter sensitivity study
outputs/               generated figures, logs, JSON results
```

## Run
```bash
pip install -r requirements.txt
python main.py --quick        # ~20 s demo
python main.py                # full run: 5 seeds x 150 generations (~5 min)
python experiments.py         # sensitivity analysis (population, mutation rate, tournament size)
```
Figures are written to `outputs/` (`01_fitness_convergence.png`, `02_paths_and_coverage_evolution.png`,
`03_coverage_heatmaps.png`, `04_best_flight_paths.png`, `05/06_ga_vs_random*.png`, `07_hyperparameter_sensitivity.png`).

## Design summary
* **Chromosome** – per drone: K=10 real (x,y) waypoints, K+1 speed multipliers, K+1 sensor-power levels, K+1 sensor ON/OFF bits.
* **Fitness** – `F = 1.0·Coverage + 0.7·Detection − 0.3·Energy − 0.2·Time − 0.45·Hazard − 6·NoFly − 3·RangeExcess`.
* **Crossover** – segment swap / geometric blend / whole-route swap (+ uniform for bits).
* **Mutation** – Gaussian shift, random reset, swap, sub-tour reversal, speed/power tweak, bit flip.
* **Constraints** – no-fly waypoints repaired to the nearest free cell; path samples in no-fly cells and range/time limit excess penalised.
* **Diversity** – adaptive mutation boost + random immigrants when mean waypoint distance < threshold.
* **Termination** – max generations or 60 generations without improvement.
