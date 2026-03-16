"""
solve.py -- Train the Monte Carlo agent and generate policy map
================================================================

Runs Monte Carlo Exploring Starts on Blackjack for 5 million
episodes, then generates:
    results/policy_map.png -- dust-style converged policy (all 360 states)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from mc_agent import MonteCarloAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NUM_EPISODES = 5_000_000
LOG_INTERVAL = 50_000


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    agent = MonteCarloAgent(seed=42)
    print(f"Training MC Exploring Starts for {NUM_EPISODES:,d} episodes...")
    curve = agent.train(NUM_EPISODES, log_interval=LOG_INTERVAL)

    hard, soft = agent.get_strategy_matrix()
    gv = agent.game_value_estimate()

    print(f"\nEstimated game value: {gv:+.4f}")
    print(f"States visited: {len(agent.policy)}")

    plot_converged_policy(hard, soft)

    print(f"\nPlots saved to {RESULTS_DIR}/")


def plot_converged_policy(hard, soft):
    """
    Dust-style scatter plot of the converged policy across all 360
    MDP states (hard + soft hands). Each cell gets 500 random points
    colored by the optimal action.
    """
    plt.rcParams.update({"font.size": 14})
    rng = np.random.default_rng(42)

    # 0=Hit(blue), 1=Stand(red), 2=Double(green)
    action_colors = {0: "#3b82f6", 1: "#ef4444", 2: "#22c55e"}
    points_per_cell = 500

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, matrix, subtitle in [
        (axes[0], hard, "Hard Hands"),
        (axes[1], soft, "Soft Hands"),
    ]:
        nrows, ncols = matrix.shape  # 18 x 10

        for i in range(nrows):
            for j in range(ncols):
                action = matrix[i][j]
                color = action_colors[action]
                px = rng.uniform(j + 0.05, j + 0.95, size=points_per_cell)
                py = rng.uniform(i + 0.05, i + 0.95, size=points_per_cell)
                ax.scatter(px, py, s=0.5, alpha=0.5, color=color,
                           rasterized=True)

        # grid lines
        for i in range(nrows + 1):
            ax.axhline(i, color="#cccccc", linewidth=0.3)
        for j in range(ncols + 1):
            ax.axvline(j, color="#cccccc", linewidth=0.3)

        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.invert_yaxis()

        col_labels = ["A"] + [str(d) for d in range(2, 11)]
        row_labels = [str(s) for s in range(4, 22)]
        ax.set_xticks([j + 0.5 for j in range(ncols)])
        ax.set_xticklabels(col_labels, fontsize=12)
        ax.set_yticks([i + 0.5 for i in range(nrows)])
        ax.set_yticklabels(row_labels, fontsize=12)

        ax.set_xlabel("Dealer Upcard", fontsize=14, labelpad=8)
        ax.set_ylabel("Player Sum", fontsize=14, labelpad=8)
        ax.set_title(subtitle, fontsize=16, fontweight="bold", pad=10)
        ax.set_aspect("equal")

    legend_elements = [
        Patch(facecolor="#3b82f6", label="Hit"),
        Patch(facecolor="#ef4444", label="Stand"),
        Patch(facecolor="#22c55e", label="Double"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=13, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Converged Policy Map: 360 MDP States after 5M Episodes",
        fontsize=18, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(os.path.join(RESULTS_DIR, "policy_map.png"), dpi=150,
                facecolor="white")
    plt.close(fig)
    print("Saved policy_map.png")


if __name__ == "__main__":
    main()
