"""
Main task synthesis orchestrator.

Provides a unified interface for generating, verifying, and saving synthesized
tasks across all domains (airline, retail, telecom). Tasks are compatible with the existing
tau2 evaluation pipeline.

Usage:
    python -m tau2.synthesis.synthesize --domain airline --num-per-template 10 --seed 42
    python -m tau2.synthesis.synthesize --domain retail --num-per-template 10 --seed 42
    python -m tau2.synthesis.synthesize --domain telecom --num-per-template 10 --seed 42
    python -m tau2.synthesis.synthesize --domain all --num-per-template 5 --seed 42
"""

import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.utils import DATA_DIR


def synthesize_domain_tasks(
    domain: str,
    num_per_template: int = 5,
    seed: int = 42,
    verify: bool = True,
) -> list[Task]:
    """
    Synthesize tasks for a specific domain.

    Args:
        domain: One of 'airline', 'retail', 'telecom'
        num_per_template: Number of tasks to generate per scenario template
        seed: Random seed for reproducibility
        verify: Whether to verify each task

    Returns:
        List of Task objects
    """
    if domain == "airline":
        from tau2.synthesis.airline_synth import create_airline_tasks
        return create_airline_tasks(num_per_template, seed, verify)
    elif domain == "retail":
        from tau2.synthesis.retail_synth import create_retail_tasks
        return create_retail_tasks(num_per_template, seed, verify)
    elif domain == "telecom":
        from tau2.synthesis.telecom_synth import create_telecom_tasks
        return create_telecom_tasks(num_per_template, seed, verify)
    else:
        raise ValueError(f"Unknown domain: {domain}. Supported: airline, retail, telecom")


def synthesize_tasks(
    domains: Optional[list[str]] = None,
    num_per_template: int = 5,
    seed: int = 42,
    verify: bool = True,
    save: bool = True,
    output_dir: Optional[str] = None,
) -> dict[str, list[Task]]:
    """
    Synthesize tasks across multiple domains.

    Args:
        domains: List of domains to synthesize for (default: all)
        num_per_template: Number of tasks per template per domain
        seed: Random seed
        verify: Whether to verify tasks
        save: Whether to save to JSON files
        output_dir: Output directory (default: data/tau2/domains/<domain>/)

    Returns:
        Dictionary mapping domain names to lists of Tasks
    """
    if domains is None:
        domains = ["airline", "retail", "telecom"]

    results = {}
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Synthesizing {domain} tasks (seed={seed}, n={num_per_template})")
        print(f"{'='*60}")

        tasks = synthesize_domain_tasks(domain, num_per_template, seed, verify)
        results[domain] = tasks

        if save:
            if output_dir:
                out_path = Path(output_dir) / f"synth_tasks_{domain}.json"
            else:
                out_path = (
                    DATA_DIR / "tau2" / "domains" / domain / "synth_tasks.json"
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump([t.model_dump() for t in tasks], f, indent=2)
            print(f"Saved {len(tasks)} tasks to {out_path}")

            # Also create a split file for the synthesized tasks
            split_path = out_path.parent / f"split_synth_tasks.json"
            task_ids = [t.id for t in tasks]
            splits = {
                "base": task_ids,
                "all": task_ids,
            }
            with open(split_path, "w") as f:
                json.dump(splits, f, indent=2)
            print(f"Saved split file to {split_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SYNTHESIS SUMMARY")
    print(f"{'='*60}")
    for domain, tasks in results.items():
        print(f"  {domain}: {len(tasks)} tasks")
        # Count by category
        categories = {}
        for t in tasks:
            cat = t.id.split("[")[0].replace("synth_", "")
            # Extract the template name (before the last _N)
            parts = cat.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                cat = parts[0]
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")
    total = sum(len(t) for t in results.values())
    print(f"  TOTAL: {total} tasks")

    return results


def main():
    parser = ArgumentParser(description="Synthesize tau2-bench tasks")
    parser.add_argument(
        "-d", "--domain",
        type=str,
        default="all",
        choices=["airline", "retail", "telecom", "all"],
        help="Domain to synthesize tasks for",
    )
    parser.add_argument(
        "-n", "--num-per-template",
        type=int,
        default=5,
        help="Number of tasks to generate per scenario template",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip task verification",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save tasks to files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for task files",
    )

    args = parser.parse_args()

    domains = ["airline", "retail", "telecom"] if args.domain == "all" else [args.domain]

    synthesize_tasks(
        domains=domains,
        num_per_template=args.num_per_template,
        seed=args.seed,
        verify=not args.no_verify,
        save=not args.no_save,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
