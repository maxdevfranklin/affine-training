"""Utilities to render validator-style summaries for local evaluation results."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tabulate import tabulate

# Environment order matches affine.setup.ENVS and validator output
ENVIRONMENT_ORDER: Tuple[str, ...] = (
    "agentgym:webshop",
    "agentgym:alfworld",
    "agentgym:babyai",
    "agentgym:sciworld",
    "agentgym:textcraft",
    "affine:sat",
    "affine:ded",
    "affine:abd",
)

# Parameters mirrored from affine.sampling.SamplingConfig
MIN_SAMPLES_PER_ENV = 200
GEOMETRIC_MEAN_GAMMA = 4.0
LAYER_SCALE = 1.0
SCI_WORLD_RANGE = (-100.0, 100.0)


@dataclass
class ValidatorSummary:
    header: List[str]
    row: List[str]

    def to_text(self) -> str:
        return tabulate([self.row], self.header, tablefmt="plain")


def format_environment_scores(per_environment: Dict[str, Dict[str, float]]) -> List[str]:
    columns, _, _ = _format_env_columns(per_environment)
    return columns


def _normalize_score(env: str, score: float) -> float:
    if env == "agentgym:sciworld":
        lo, hi = SCI_WORLD_RANGE
        if hi == lo:
            return 0.0
        if 0.0 <= score <= 1.0:
            return float(score)
        return (score - lo) / (hi - lo)
    return float(score)


def _format_env_columns(per_env: Dict[str, Dict[str, float]]) -> Tuple[List[str], Dict[str, float], Dict[str, int]]:
    formatted = []
    mean_scores: Dict[str, float] = {}
    samples: Dict[str, int] = {}

    for env_name in ENVIRONMENT_ORDER:
        metrics = per_env.get(env_name, {})
        mean_score = metrics.get("mean_score", 0.0)
        ci = metrics.get("confidence_interval", {})
        ci_lower = ci.get("lower", 0.0)
        ci_upper = ci.get("upper", 0.0)
        num_samples = metrics.get("num_samples", 0)

        formatted.append(
            f"{mean_score * 100:.2f}/[{ci_lower * 100:.2f},{ci_upper * 100:.2f}]/{num_samples}"
        )

        mean_scores[env_name] = mean_score
        samples[env_name] = num_samples

    return formatted, mean_scores, samples


def _compute_layer_points(mean_scores: Dict[str, float], samples: Dict[str, int]) -> Dict[int, float]:
    n_envs = len(ENVIRONMENT_ORDER)
    layer_points = {s: 0.0 for s in range(1, n_envs + 1)}

    layer_weights = {1: LAYER_SCALE}
    for s in range(2, n_envs + 1):
        layer_weights[s] = LAYER_SCALE * (2 ** s)

    for subset_size in range(1, n_envs + 1):
        for subset in itertools.combinations(ENVIRONMENT_ORDER, subset_size):
            if any(samples.get(env, 0) < MIN_SAMPLES_PER_ENV for env in subset):
                continue

            normalized_scores = [
                max(0.0, min(1.0, _normalize_score(env, mean_scores.get(env, 0.0))))
                for env in subset
            ]

            if not normalized_scores or any(score <= 0.0 for score in normalized_scores):
                continue

            geometric_mean = math.prod(normalized_scores) ** (1.0 / len(normalized_scores))
            comprehensive_score = geometric_mean ** GEOMETRIC_MEAN_GAMMA
            layer_points[subset_size] += layer_weights[subset_size] * comprehensive_score

    return layer_points


def build_validator_summary(
    *,
    per_environment: Dict[str, Dict[str, float]],
    model_name: str,
    uid: int = 0,
    revision: str = "local",
    first_block: int = 0,
):
    env_columns, mean_scores, samples = _format_env_columns(per_environment)
    layer_points = _compute_layer_points(mean_scores, samples)
    pts = sum(layer_points.values())

    eligible = all(samples.get(env, 0) >= MIN_SAMPLES_PER_ENV for env in ENVIRONMENT_ORDER)
    avg_score = (
        sum(
            max(0.0, min(1.0, _normalize_score(env, mean_scores.get(env, 0.0))))
            for env in ENVIRONMENT_ORDER
        )
        / len(ENVIRONMENT_ORDER)
        if ENVIRONMENT_ORDER
        else 0.0
    )

    header = (
        ["UID", "Model", "Rev"]
        + list(ENVIRONMENT_ORDER)
        + [f"L{s}" for s in range(1, len(ENVIRONMENT_ORDER) + 1)]
        + ["Pts", "Elig", "FirstBlk", "Wgt", "Avg"]
    )

    row = [
        str(uid),
        model_name,
        revision[:5],
        *env_columns,
        *[f"{layer_points[s]:.1f}" for s in range(1, len(ENVIRONMENT_ORDER) + 1)],
        f"{pts:.2f}",
        "Y" if eligible else "N",
        str(first_block),
        f"{1.0 if eligible and pts > 0 else 0.0:.4f}",
        f"{avg_score * 100:.2f}",
    ]

    return ValidatorSummary(header=header, row=row)

