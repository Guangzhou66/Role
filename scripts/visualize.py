"""
Visualization module for all experiments.

Generates publication-quality figures for:
- A1: Cheap-vs-full gap bar charts
- A2: Budget sensitivity curves with degradation slopes
- A3: Difficulty-stratified failure distributions
- B1: Packet utility heatmaps and temporal distributions
- B2: Tail dominance regret charts
- B3: Role sensitivity matrices
- B5: Receiver brittleness tables
- B6: Distribution shift visualizations (PCA/t-SNE)
- C1: Method alignment comparison bars
- C3: Main results tables
- C4: Pareto curves (Accuracy vs Budget, Accuracy vs Latency)
- C5: Oracle upper bound gap charts
- C6: Error bars with confidence intervals
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from src.utils.helpers import ensure_dir, load_json


COLORS = {
    "full": "#2ecc71",
    "recency": "#e74c3c",
    "fixed_truncation": "#e67e22",
    "uniform": "#9b59b6",
    "random": "#95a5a6",
    "role_aware": "#3498db",
    "recency+adapter": "#e74c3c",
    "role_aware+adapter": "#2980b9",
    "oracle": "#f1c40f",
}

MARKERS = {
    "full": "o",
    "recency": "s",
    "fixed_truncation": "D",
    "uniform": "^",
    "random": "v",
    "role_aware": "P",
    "recency+adapter": "X",
    "role_aware+adapter": "*",
    "oracle": "h",
}


def _check_plot():
    if not HAS_PLOT:
        raise RuntimeError("matplotlib/seaborn required for visualization")


# ════════════════════════════════════════════════
#  A1: Cheap-vs-Full Gap
# ════════════════════════════════════════════════

def plot_a1_gap(data: Dict, output_dir: str = "figures"):
    """Bar chart of cheap-full gap across tasks."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = [k for k in data if k != "summary"]
    strategies = ["recency", "fixed_truncation"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    width = 0.35

    for i, strat in enumerate(strategies):
        gaps = []
        for task in tasks:
            gap = data[task].get(strat, {}).get("gap", 0)
            gaps.append(gap)
        ax.bar(x + i * width, gaps, width, label=strat.replace("_", " ").title(),
               color=COLORS.get(strat, "#333"))

    ax.set_xlabel("Task")
    ax.set_ylabel("Performance Gap (Δ)")
    ax.set_title("A1: Cheap-vs-Full Communication Gap")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "a1_cheap_full_gap.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "a1_cheap_full_gap.png"), dpi=200)
    plt.close(fig)


# ════════════════════════════════════════════════
#  A2: Budget Sensitivity Curves
# ════════════════════════════════════════════════

def plot_a2_budget_curves(data: Dict, output_dir: str = "figures"):
    """Line plots of performance vs budget for each task."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = [k for k in data if k != "summary"]

    for task in tasks:
        fig, ax = plt.subplots(figsize=(8, 5))
        task_data = data[task]

        for strat_name in ["recency", "fixed_truncation"]:
            if strat_name not in task_data:
                continue
            curve = task_data[strat_name].get("budget_curve", {})
            budgets = sorted([float(b) for b in curve.keys()])
            perfs = [curve[str(b)]["accuracy"] for b in budgets]

            ax.plot(budgets, perfs,
                    marker=MARKERS.get(strat_name, "o"),
                    color=COLORS.get(strat_name, "#333"),
                    label=strat_name.replace("_", " ").title(),
                    linewidth=2, markersize=8)

        if "full" in task_data:
            full_perf = task_data["full"]["accuracy"]
            ax.axhline(y=full_perf, color=COLORS["full"],
                        linestyle="--", linewidth=2, label="Full")

        ax.set_xlabel("Communication Budget (b)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"A2: Budget Sensitivity — {task}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"a2_budget_{task}.pdf"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"a2_budget_{task}.png"), dpi=200)
        plt.close(fig)


# ════════════════════════════════════════════════
#  A3: Difficulty-Stratified Failure
# ════════════════════════════════════════════════

def plot_a3_difficulty(data: Dict, output_dir: str = "figures"):
    """Grouped bar chart of fail rates by difficulty group."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())
    fig, axes = plt.subplots(1, min(len(tasks), 3), figsize=(15, 5),
                              sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks[:3]):
        task_data = data[task]
        groups = ["easy", "medium", "hard"]

        for strat in ["recency", "fixed_truncation"]:
            if strat not in task_data:
                continue
            gm = task_data[strat].get("group_metrics", {})
            frs = [gm.get(g, {}).get("fail_rate", 0) for g in groups]
            x = np.arange(len(groups))
            offset = 0 if strat == "recency" else 0.35
            ax.bar(x + offset, frs, 0.35,
                   color=COLORS.get(strat, "#333"),
                   label=strat.replace("_", " ").title())

        ax.set_xticks(np.arange(len(groups)) + 0.175)
        ax.set_xticklabels(groups)
        ax.set_title(task)
        ax.set_ylabel("Failure Rate")

    axes[0].legend()
    fig.suptitle("A3: Failure Rate by Difficulty Group", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "a3_difficulty_stratified.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "a3_difficulty_stratified.png"), dpi=200)
    plt.close(fig)


# ════════════════════════════════════════════════
#  B1: Temporal Utility Heatmap
# ════════════════════════════════════════════════

def plot_b1_temporal_utility(data: Dict, output_dir: str = "figures"):
    """Bar chart showing utility distribution across head/mid/tail."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(tasks))
    width = 0.25
    segments = ["U_head", "U_mid", "U_tail"]
    seg_colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, seg in enumerate(segments):
        vals = [data[t].get("segment_avg_utility", {}).get(seg, 0)
                for t in tasks]
        ax.bar(x + i * width, vals, width,
               label=seg.replace("U_", "").title(),
               color=seg_colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylabel("Average Utility")
    ax.set_title("B1: Temporal Utility Distribution (Head / Mid / Tail)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "b1_temporal_utility.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "b1_temporal_utility.png"), dpi=200)
    plt.close(fig)


# ════════════════════════════════════════════════
#  B2: Tail Dominance Regret
# ════════════════════════════════════════════════

def plot_b2_tail_regret(data: Dict, output_dir: str = "figures"):
    """Bar chart comparing tail vs uniform vs oracle."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(tasks))
    width = 0.25

    for i, method in enumerate(["tail", "uniform", "oracle"]):
        perfs = [data[t].get(method, {}).get("accuracy", 0) for t in tasks]
        ax.bar(x + i * width, perfs, width,
               label=method.title(), color=list(COLORS.values())[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("B2: Tail-Only vs Uniform vs Oracle Selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "b2_tail_dominance.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "b2_tail_dominance.png"), dpi=200)
    plt.close(fig)


# ════════════════════════════════════════════════
#  B3: Role Sensitivity Matrix
# ════════════════════════════════════════════════

def plot_b3_role_sensitivity(data: Dict, output_dir: str = "figures"):
    """Heatmap of role sensitivity across tasks."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())
    roles = ["planner", "critic", "refiner"]

    matrix = []
    for task in tasks:
        row = []
        sens = data[task].get("role_sensitivity", {})
        for role in roles:
            row.append(sens.get(role, {}).get("delta_role", 0))
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f",
                xticklabels=[r.title() for r in roles],
                yticklabels=tasks,
                cmap="YlOrRd", ax=ax)
    ax.set_title("B3: Role Sensitivity Matrix (Δ_r^role)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "b3_role_sensitivity.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "b3_role_sensitivity.png"), dpi=200)
    plt.close(fig)


# ════════════════════════════════════════════════
#  C1: Method Alignment
# ════════════════════════════════════════════════

def plot_c1_method_alignment(data: Dict, output_dir: str = "figures"):
    """Grouped bar chart of all 5 configurations."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())
    methods = ["full", "recency", "role_aware", "recency+adapter",
               "role_aware+adapter"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tasks))
    width = 0.15

    for i, method in enumerate(methods):
        perfs = [
            data[t].get("performances", {}).get(method, 0) for t in tasks
        ]
        ax.bar(x + i * width, perfs, width,
               label=method.replace("+", " + ").replace("_", " ").title(),
               color=COLORS.get(method, f"C{i}"))

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("C1: Method-to-Problem Alignment")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "c1_method_alignment.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "c1_method_alignment.png"), dpi=200)
    plt.close(fig)


# ════════════════════════════════════════════════
#  C4: Pareto Curves
# ════════════════════════════════════════════════

def plot_c4_pareto(data: Dict, output_dir: str = "figures"):
    """Accuracy vs budget Pareto curves for each task."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())

    for task in tasks:
        fig, ax = plt.subplots(figsize=(8, 5))
        task_data = data[task]

        for method, info in task_data.items():
            curve = info.get("budget_curve", {})
            if not curve:
                continue
            budgets = sorted([float(b) for b in curve.keys()])
            perfs = [curve[str(b)] for b in budgets]

            ax.plot(budgets, perfs,
                    marker=MARKERS.get(method, "o"),
                    color=COLORS.get(method, "#333"),
                    label=method.replace("+", " + ").replace("_", " ").title(),
                    linewidth=2, markersize=8)

        ax.set_xlabel("Communication Budget")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"C4: Budget-Performance Pareto — {task}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"c4_pareto_{task}.pdf"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"c4_pareto_{task}.png"), dpi=200)
        plt.close(fig)


# ════════════════════════════════════════════════
#  C6: Statistical Reliability
# ════════════════════════════════════════════════

def plot_c6_reliability(data: Dict, output_dir: str = "figures"):
    """Error bar chart with confidence intervals."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())

    for task in tasks:
        fig, ax = plt.subplots(figsize=(8, 5))
        task_data = data[task]

        methods = list(task_data.keys())
        means = [task_data[m].get("mean", 0) for m in methods]
        stds = [task_data[m].get("std", 0) for m in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[COLORS.get(m, "#3498db") for m in methods],
                      alpha=0.85)

        # Add CI whiskers
        for i, m in enumerate(methods):
            ci = task_data[m].get("ci_95", [0, 0])
            ax.plot([i, i], ci, color="black", linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("+", "+\n").replace("_", " ") for m in methods],
            fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"C6: Statistical Reliability — {task}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"c6_reliability_{task}.pdf"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"c6_reliability_{task}.png"), dpi=200)
        plt.close(fig)


# ════════════════════════════════════════════════
#  D1: Length Scaling
# ════════════════════════════════════════════════

def plot_d1_length_scaling(data: Dict, output_dir: str = "figures"):
    """Line chart of advantage vs communication length."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())

    for task in tasks:
        fig, ax = plt.subplots(figsize=(8, 5))
        task_data = data[task]

        rounds = sorted([
            int(k.split("_")[1])
            for k in task_data if k.startswith("rounds_")
        ])

        for method in ["recency", "role_aware", "role_aware+adapter"]:
            perfs = []
            for r in rounds:
                key = f"rounds_{r}"
                perf = task_data.get(key, {}).get(method, {}).get("accuracy", 0)
                perfs.append(perf)

            ax.plot(rounds, perfs,
                    marker=MARKERS.get(method, "o"),
                    color=COLORS.get(method, "#333"),
                    label=method.replace("+", " + ").replace("_", " ").title(),
                    linewidth=2, markersize=8)

        ax.set_xlabel("Number of Rounds (Communication Length)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"D1: Length Scaling — {task}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"d1_length_{task}.pdf"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"d1_length_{task}.png"), dpi=200)
        plt.close(fig)


# ════════════════════════════════════════════════
#  D2: Agent Count Scaling
# ════════════════════════════════════════════════

def plot_d2_agent_scaling(data: Dict, output_dir: str = "figures"):
    """Bar chart of performance by agent topology."""
    _check_plot()
    ensure_dir(output_dir)

    tasks = list(data.keys())

    for task in tasks:
        fig, ax = plt.subplots(figsize=(10, 5))
        task_data = data[task]
        topos = ["2-agent", "3-agent", "4-agent"]
        methods = ["full", "recency", "role_aware", "role_aware+adapter"]

        x = np.arange(len(topos))
        width = 0.2

        for i, method in enumerate(methods):
            perfs = [
                task_data.get(t, {}).get(method, {}).get("accuracy", 0)
                for t in topos
            ]
            ax.bar(x + i * width, perfs, width,
                   label=method.replace("+", " + ").replace("_", " ").title(),
                   color=COLORS.get(method, f"C{i}"))

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(topos)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"D2: Agent Count Scaling — {task}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"d2_agent_scaling_{task}.pdf"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"d2_agent_scaling_{task}.png"), dpi=200)
        plt.close(fig)


# ════════════════════════════════════════════════
#  Generate All Figures
# ════════════════════════════════════════════════

def generate_all_figures(results_dir: str = "results",
                         output_dir: str = "figures"):
    """Generate all figures from saved experiment results."""
    _check_plot()
    ensure_dir(output_dir)

    plot_fns = {
        "exp_a/a1_cheap_full_gap.json": plot_a1_gap,
        "exp_a/a2_budget_sensitivity.json": plot_a2_budget_curves,
        "exp_a/a3_difficulty_stratified.json": plot_a3_difficulty,
        "exp_b/b1_temporal_utility_distribution.json": plot_b1_temporal_utility,
        "exp_b/b2_tail_dominance_test.json": plot_b2_tail_regret,
        "exp_b/b3_role_sensitivity_matrix.json": plot_b3_role_sensitivity,
        "exp_c/c1_method_to_problem_alignment.json": plot_c1_method_alignment,
        "exp_c/c4_budget_performance_pareto.json": plot_c4_pareto,
        "exp_c/c6_statistical_reliability.json": plot_c6_reliability,
        "exp_d/d1_length_scaling.json": plot_d1_length_scaling,
        "exp_d/d2_agent_count_scaling.json": plot_d2_agent_scaling,
    }

    for filename, plot_fn in plot_fns.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            try:
                data = load_json(path)
                plot_fn(data, output_dir)
                print(f"Generated figure for {filename}")
            except Exception as e:
                print(f"Failed to generate figure for {filename}: {e}")
        else:
            print(f"Skipping {filename} (not found)")


if __name__ == "__main__":
    generate_all_figures()
