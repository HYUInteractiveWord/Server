import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import (
    align_pitch_by_first_voiced,
    align_contours_by_cross_correlation,
)


def _ensure_parent_dir(output_path: str):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_single_pitch_graph(times, f0, title: str, label: str, output_path: str) -> str:
    _ensure_parent_dir(output_path)

    plt.figure(figsize=(12, 4))
    plt.plot(times, f0, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_overlay_pitch_graph(t_ref, f0_ref, t_usr, f0_usr, output_path: str) -> str:
    """
    Pitch Compare (Original)
    """
    _ensure_parent_dir(output_path)

    plt.figure(figsize=(12, 5))
    plt.plot(t_ref, f0_ref, label="TTS Pitch")
    plt.plot(t_usr, f0_usr, label="User Pitch", alpha=0.85)
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title("Pitch Comparison (Original)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_alignment_summary_graph(t_ref, f0_ref, t_usr, f0_usr, output_path: str) -> str:
    """
    하나의 PNG 안에:
    1) First Voiced Alignment
    2) Cross Correlation Alignment
    두 개를 같이 그림
    """
    _ensure_parent_dir(output_path)

    # First aligned
    t_ref2, f0_ref2, _ = align_pitch_by_first_voiced(t_ref, f0_ref)
    t_usr2, f0_usr2, _ = align_pitch_by_first_voiced(t_usr, f0_usr)

    # Cross correlation
    ref_n, usr_n, usr_shifted, shift = align_contours_by_cross_correlation(
        f0_ref,
        f0_usr,
    )
    x = np.linspace(0, 1, len(ref_n))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # 1) First aligned
    axes[0].plot(t_ref2, f0_ref2, label="TTS Pitch (aligned)")
    axes[0].plot(t_usr2, f0_usr2, label="User Pitch (aligned)", alpha=0.85)
    axes[0].set_xlabel("Aligned Time (s)")
    axes[0].set_ylabel("F0 (Hz)")
    axes[0].set_title("Pitch Comparison (First Voiced Align)")
    axes[0].legend()
    axes[0].grid(True)

    # 2) Cross correlation
    axes[1].plot(x, ref_n, label="TTS Pitch")
    axes[1].plot(x, usr_n, label="User Before", alpha=0.4)
    axes[1].plot(x, usr_shifted, label=f"User Align shift={shift}", linewidth=2)
    axes[1].set_xlabel("Normalized Time")
    axes[1].set_ylabel("Normalized F0")
    axes[1].set_title("Pitch Comparison (Cross Correlation)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_score_summary_graph(scores: dict, output_path: str) -> str:
    """
    각 항목 + 최종점수를 막대그래프로 저장
    """
    _ensure_parent_dir(output_path)

    labels = [
        "Pronunciation",
        "Phoneme",
        "Pitch",
        "Duration",
        "Final",
    ]
    values = [
        scores.get("pronunciation_score", 0.0),
        scores.get("phoneme_score", 0.0),
        scores.get("pitch_score", 0.0),
        scores.get("duration_score", 0.0),
        scores.get("final_score", 0.0),
    ]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bars = ax.bar(labels, values)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("Score Summary")
    ax.grid(axis="y", alpha=0.3)

    _add_value_labels(ax, bars)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path