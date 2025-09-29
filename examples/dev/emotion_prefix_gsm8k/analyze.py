#!/usr/bin/env python3
"""Analyze Emotion Prefix GSM8K results (works for rollouts and non-rollouts outputs).

Usage:
  python examples/emotion_prefix_gsm8k/analyze.py <results_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_results(exp_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "variants": {},
        "total": 0,
        "correct": 0,
    }
    # Iterate variant folders
    for variant_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
        variant = variant_dir.name
        if variant in {"agent_states", "trajectories"}:  # rollouts summary dirs
            continue
        v_total = 0
        v_correct = 0
        for sample_dir in sorted([p for p in variant_dir.iterdir() if p.is_dir()]):
            sj = sample_dir / "sample.json"
            if not sj.exists():
                continue
            try:
                data = json.loads(sj.read_text())
            except Exception:
                continue
            ok = False
            # Rollouts path: metrics.correctness in [0,1]
            if isinstance(data.get("metrics"), dict):
                ok = float(data["metrics"].get("correctness", 0.0)) > 0.5
            # Non-rollouts path: boolean 'correct'
            elif isinstance(data.get("correct"), bool):
                ok = bool(data["correct"]) 
            v_total += 1
            v_correct += 1 if ok else 0
        if v_total > 0:
            summary["variants"][variant] = {
                "total": v_total,
                "correct": v_correct,
                "accuracy": v_correct / v_total,
            }
            summary["total"] += v_total
            summary["correct"] += v_correct
    if summary["total"] > 0:
        summary["accuracy"] = summary["correct"] / summary["total"]
    else:
        summary["accuracy"] = 0.0
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("🔍 Emotion Prefix GSM8K — Results")
    print("=" * 60)
    print(f"Overall: {summary['correct']}/{summary['total']}  (acc={summary['accuracy']:.3f})")
    print()
    print("Per-variant:")
    for variant in sorted(summary["variants"].keys()):
        v = summary["variants"][variant]
        print(f"  {variant:>14}: {v['correct']:>3}/{v['total']:<3}  (acc={v['accuracy']:.3f})")


def main():
    if len(sys.argv) != 2:
        print("Usage: python examples/emotion_prefix_gsm8k/analyze.py <results_dir>")
        sys.exit(2)
    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"❌ Not found: {exp_dir}")
        sys.exit(1)
    try:
        summary = load_results(exp_dir)
        print_summary(summary)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

