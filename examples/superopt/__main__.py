"""CLI entry point: python -m examples.superopt [target] [options]"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Superoptimizer for gint bytecodes — "
        "find shorter equivalent instruction sequences.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Target to optimize (e.g. relu, abs, gelu). "
        "Use --list to see all targets.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available targets and exit.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run search on all targets.",
    )
    parser.add_argument(
        "--max-len", type=int, default=None,
        help="Maximum candidate sequence length (default: target_len - 1).",
    )
    parser.add_argument(
        "--mode", choices=["brute", "stochastic"], default="brute",
        help="Search mode (default: brute).",
    )
    parser.add_argument(
        "--stochastic-length", type=int, default=None,
        help="Candidate length for stochastic mode.",
    )
    parser.add_argument(
        "--stochastic-pop", type=int, default=100000,
        help="Population per generation for stochastic mode.",
    )
    parser.add_argument(
        "--stochastic-gens", type=int, default=100,
        help="Number of generations for stochastic mode.",
    )

    args = parser.parse_args()

    if args.list:
        from .targets import list_targets
        print("Available targets:")
        list_targets()
        return

    if args.all:
        from .targets import TARGETS
        from .search import brute_force
        print("=" * 60)
        print("Running brute-force search on all targets")
        print("=" * 60)
        results = {}
        for name in TARGETS:
            print()
            print("-" * 60)
            matches = brute_force(name, max_length=args.max_len)
            results[name] = matches
            print()

        print("=" * 60)
        print("Summary:")
        for name, matches in results.items():
            t = TARGETS[name]
            ref_len = len(t["body"])
            if matches:
                best = len(matches[0])
                print(f"  {name:20s}  {ref_len} -> {best} insns  IMPROVED")
            else:
                print(f"  {name:20s}  {ref_len} insns  optimal")
        return

    if args.target is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "brute":
        from .search import brute_force
        brute_force(args.target, max_length=args.max_len)
    else:
        from .search import stochastic
        length = args.stochastic_length
        if length is None:
            from .targets import get_target
            length = len(get_target(args.target)["body"])
        stochastic(
            args.target,
            length=length,
            n_candidates=args.stochastic_pop,
            n_generations=args.stochastic_gens,
        )


if __name__ == "__main__":
    main()
