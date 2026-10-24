"""All plotting helpers (matplotlib, headless)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

COLORS = ["#d62728", "#1f77b4", "#2ca02c"]


def _draw_env(ax, env, show_priority=True):
    n = env.cfg.grid_size
    if show_priority:
        ax.imshow(env.priority, origin="lower", cmap="YlOrBr", alpha=0.75, vmin=0, vmax=1)
    hz = np.ma.masked_less(env.hazard, 0.25)
    ax.imshow(hz, origin="lower", cmap="Purples", alpha=0.55, vmin=0, vmax=1.2)
    ax.imshow(np.ma.masked_where(~env.nofly, env.nofly), origin="lower", cmap="gray", alpha=0.9)
    ax.scatter(env.targets[:, 0], env.targets[:, 1], marker="x", c="k", s=14, lw=0.9, zorder=3)
    ax.plot(*env.cfg.base, "s", c="lime", mec="k", ms=9, zorder=5)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5); ax.set_xticks([]); ax.set_yticks([])


def plot_environment(env, path):
    fig, ax = plt.subplots(figsize=(6, 6)); _draw_env(ax, env)
    ax.set_title("Hazard grid: priority zones (orange), hazard pockets (purple),\nno-fly (black), targets (x), base (green)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _draw_paths(ax, env, c, title):
    _draw_env(ax, env)
    for d, r in enumerate(c.routes(env.cfg.base)):
        ax.plot(r[:, 0], r[:, 1], "-o", c=COLORS[d % 3], ms=3, lw=1.6, alpha=0.95, label=f"UAV {d+1}")
    ax.set_title(title, fontsize=9)


def plot_paths(env, c, path, title="Evolved UAV flight paths"):
    fig, ax = plt.subplots(figsize=(6.2, 6)); _draw_paths(ax, env, c, title)
    ax.legend(loc="upper right", fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_evolution(env, snaps, path):
    """Grid of paths + coverage heat-maps for chosen generations."""
    gens = sorted(snaps); k = len(gens)
    fig, axes = plt.subplots(2, k, figsize=(3.3 * k, 6.8))
    for j, g in enumerate(gens):
        c = snaps[g]
        _draw_paths(axes[0, j], env, c, f"Gen {g}  F={c.fitness:.3f}\ncov={c.info['coverage']:.2f} det={c.info['detection']:.2f}")
        ax = axes[1, j]
        ax.imshow(env.priority, origin="lower", cmap="Greys", alpha=0.35)
        ax.imshow(np.ma.masked_where(~c.info["covered_mask"], c.info["covered_mask"].astype(float)), origin="lower",
                  cmap="Reds", alpha=0.75, vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~env.nofly, env.nofly), origin="lower", cmap="gray", alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Sensor coverage (red) over priority map", fontsize=8)
    fig.suptitle("Evolution of best flight plan and coverage across generations", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_heatmap_progress(env, snaps, path):
    gens = sorted(snaps); k = len(gens)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.5))
    for ax, g in zip(np.atleast_1d(axes), gens):
        c = snaps[g]
        m = c.info["covered_mask"].astype(float) * env.priority
        im = ax.imshow(m, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"Gen {g}: covered priority\n{c.info['coverage']*100:.0f}% of total", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.015, label="priority value covered")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_convergence(hist, path, rs_hist=None):
    g = np.arange(len(hist["best"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(g, hist["best"], c="#d62728", lw=2, label="Best fitness")
    ax[0].plot(g, hist["avg"], c="#1f77b4", lw=1.6, label="Average fitness")
    ax[0].fill_between(g, hist["worst"], hist["best"], color="#1f77b4", alpha=0.08, label="Best-worst range")
    lo = min(hist["best"]) - 0.5
    ax[0].set_ylim(max(min(hist["avg"]), lo - 3), max(hist["best"]) + 0.3)
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness"); ax[0].set_title("Fitness convergence"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(g, hist["diversity"], c="#2ca02c", lw=2, label="Population diversity (cells)")
    ax[1].set_xlabel("Generation"); ax[1].set_ylabel("Mean waypoint distance"); ax[1].grid(alpha=.3)
    a2 = ax[1].twinx(); a2.plot(g, hist["mut_rate"], c="#ff7f0e", ls="--", label="Mutation rate"); a2.set_ylabel("Mutation rate")
    ax[1].set_title("Diversity & adaptive mutation")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_vs_random(ga_best_curves, rs_curves, path, env=None, ga_best=None, rs_best=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (e, f) in enumerate(ga_best_curves):
        ax.plot(e, f, c="#d62728", alpha=.7, label="GA (each seed)" if i == 0 else None)
    for i, (e, f) in enumerate(rs_curves):
        ax.plot(e, f, c="#7f7f7f", alpha=.8, label="Random search (each seed)" if i == 0 else None)
    ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best fitness so far"); ax.grid(alpha=.3)
    ax.set_title("GA vs random search (equal evaluation budget)"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_compare_paths(env, ga_best, rs_best, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    _draw_paths(ax[0], env, ga_best, f"GA best  F={ga_best.fitness:.3f}  cov={ga_best.info['coverage']:.2f}  feasible={ga_best.info['feasible']}")
    _draw_paths(ax[1], env, rs_best, f"Random-search best  F={rs_best.fitness:.3f}  cov={rs_best.info['coverage']:.2f}  feasible={rs_best.info['feasible']}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sensitivity(results, path):
    """results: dict param -> dict(label -> list of final best fitness)."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8), sharey=True)
    for ax, (name, d) in zip(np.atleast_1d(axes), results.items()):
        labels = list(d); m = [np.mean(d[l]) for l in labels]; s = [np.std(d[l]) for l in labels]
        ax.bar(range(len(labels)), m, yerr=s, color="#4c78a8", capsize=4)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_title(name, fontsize=10); ax.grid(axis="y", alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("Final best fitness (mean ± std)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
