"""
Advanced analysis metrics for Experiment B (failure mechanism analysis).

Covers:
- B1: Packet utility (leave-one-out)
- B3: Role sensitivity matrix
- B4: Sender-receiver utility asymmetry
- B5: Receiver brittleness
- B6: Communication distribution shift
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

from config import AgentRole
from src.agents.base import CommPacket


# ──────────────────────────────────────────────
#  B1: Temporal utility (leave-one-packet-out)
# ──────────────────────────────────────────────

def compute_packet_utility(
    packets: List[CommPacket],
    eval_fn: Callable[[List[CommPacket]], float],
    full_perf: Optional[float] = None,
) -> List[Dict]:
    """
    Leave-one-packet-out utility analysis.
    U_p = Perf(full) - Perf(full \\ p)

    eval_fn: takes a list of packets, returns performance score.
    Returns list of dicts with packet info + utility.
    """
    if full_perf is None:
        full_perf = eval_fn(packets)

    results = []
    for i, pkt in enumerate(packets):
        remaining = packets[:i] + packets[i + 1:]
        perf_without = eval_fn(remaining)
        utility = full_perf - perf_without

        results.append({
            "packet_idx": i,
            "agent_role": pkt.agent_role.value,
            "step_idx": pkt.step_idx,
            "utility": utility,
            "position_ratio": i / max(len(packets) - 1, 1),
        })
        pkt.utility = utility

    return results


def temporal_utility_summary(
    utility_results: List[Dict], n_segments: int = 3
) -> Dict[str, float]:
    """
    Compute average utility for head / mid / tail segments.
    Corresponds to Ū_head, Ū_mid, Ū_tail from document.
    """
    n = len(utility_results)
    seg_size = max(n // n_segments, 1)

    segment_names = ["head", "mid", "tail"] if n_segments == 3 else \
                    [f"seg_{i}" for i in range(n_segments)]
    summary = {}
    for i, name in enumerate(segment_names):
        start = i * seg_size
        end = min((i + 1) * seg_size, n) if i < n_segments - 1 else n
        seg_utils = [r["utility"] for r in utility_results[start:end]]
        summary[f"U_{name}"] = np.mean(seg_utils) if seg_utils else 0.0

    return summary


# ──────────────────────────────────────────────
#  B3: Role sensitivity matrix
# ──────────────────────────────────────────────

def compute_role_sensitivity(
    packets: List[CommPacket],
    eval_fn: Callable[[List[CommPacket]], float],
    full_perf: Optional[float] = None,
    roles: Optional[List[AgentRole]] = None,
) -> Dict[str, float]:
    """
    Δ_r^role = Perf(full) - Perf(compress role r)

    Compresses all packets from one role at a time while keeping
    others at full, then measures the performance drop.
    """
    if full_perf is None:
        full_perf = eval_fn(packets)
    if roles is None:
        roles = list(set(p.agent_role for p in packets))

    sensitivity = {}
    for role in roles:
        remaining = [p for p in packets if p.agent_role != role]
        perf_compressed = eval_fn(remaining)
        delta = full_perf - perf_compressed
        sensitivity[role.value] = delta

    return sensitivity


# ──────────────────────────────────────────────
#  B4: Sender-receiver utility asymmetry
# ──────────────────────────────────────────────

def compute_sender_receiver_utility(
    packets: List[CommPacket],
    eval_fn_per_receiver: Dict[AgentRole, Callable[[List[CommPacket]], float]],
    full_perf_per_receiver: Optional[Dict[AgentRole, float]] = None,
) -> Dict[str, Dict]:
    """
    U_p^{i→j} = Perf_j(full) - Perf_j(full \\ p)

    eval_fn_per_receiver: {receiver_role: eval_fn}
    Returns nested dict: {packet_idx: {receiver_role: utility, ...}}
    """
    receivers = list(eval_fn_per_receiver.keys())
    if full_perf_per_receiver is None:
        full_perf_per_receiver = {
            r: fn(packets) for r, fn in eval_fn_per_receiver.items()
        }

    results = {}
    for i, pkt in enumerate(packets):
        remaining = packets[:i] + packets[i + 1:]
        pkt_result = {
            "sender": pkt.agent_role.value,
            "step_idx": pkt.step_idx,
        }
        utilities = []
        for recv_role in receivers:
            fn = eval_fn_per_receiver[recv_role]
            perf_without = fn(remaining)
            u = full_perf_per_receiver[recv_role] - perf_without
            pkt_result[recv_role.value] = u
            utilities.append(u)

        pkt_result["mean_utility"] = float(np.mean(utilities))
        pkt_result["var_utility"] = float(np.var(utilities))
        results[f"packet_{i}"] = pkt_result

    return results


# ──────────────────────────────────────────────
#  B5: Receiver brittleness
# ──────────────────────────────────────────────

def compute_receiver_brittleness(
    perf_full: float,
    perf_compressed: float,
    predictions_compressed: Optional[List] = None,
    predictions_full: Optional[List] = None,
    rankings_compressed: Optional[List] = None,
    rankings_full: Optional[List] = None,
) -> Dict[str, float]:
    """
    B^recv = Perf(full) - Perf(compressed)
    C^teacher = mean(1[y_compressed == y_full])
    ρ = Spearman(r_compressed, r_full)
    """
    result = {
        "brittleness": perf_full - perf_compressed,
    }

    if predictions_compressed is not None and predictions_full is not None:
        n = len(predictions_compressed)
        consistency = sum(
            c == f for c, f in zip(predictions_compressed, predictions_full)
        ) / max(n, 1)
        result["teacher_consistency"] = consistency

    if rankings_compressed is not None and rankings_full is not None:
        rho, pval = spearmanr(rankings_compressed, rankings_full)
        result["spearman_rho"] = float(rho)
        result["spearman_pval"] = float(pval)

    return result


# ──────────────────────────────────────────────
#  B6: Communication distribution shift
# ──────────────────────────────────────────────

def compute_distribution_shift(
    embeddings_full: np.ndarray,
    embeddings_compressed: np.ndarray,
    use_mmd: bool = False,
) -> Dict[str, float]:
    """
    Measure distribution shift between full and compressed comm embeddings.

    D_cos(μ^a, μ^b) = 1 - cos_sim(μ^a, μ^b)
    Optionally: MMD^2(P, Q)
    """
    mu_full = embeddings_full.mean(axis=0)
    mu_comp = embeddings_compressed.mean(axis=0)

    cos_d = float(cosine_dist(mu_full, mu_comp))
    l2_d = float(np.linalg.norm(mu_full - mu_comp))

    result = {
        "cosine_distance": cos_d,
        "l2_distance": l2_d,
    }

    if use_mmd:
        result["mmd2"] = _compute_mmd2(embeddings_full, embeddings_compressed)

    return result


def _compute_mmd2(X: np.ndarray, Y: np.ndarray,
                  sigma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy (squared) with Gaussian kernel.
    MMD^2(P,Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    """
    def rbf(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * sigma ** 2))

    kxx = rbf(X, X).mean()
    kyy = rbf(Y, Y).mean()
    kxy = rbf(X, Y).mean()
    return float(kxx + kyy - 2 * kxy)
