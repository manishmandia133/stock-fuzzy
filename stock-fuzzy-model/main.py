"""Entry point.

Examples:
    python main.py                              # all variants, 10 seeds
    python main.py --variants Fuzzy_MoE Fuzzy_Input --seeds 5
    python main.py --variants Fuzzy_ANFIS --seeds 30 --no-plots
"""
import argparse

import config as cfg
from run_all import main


def _parse():
    p = argparse.ArgumentParser(description="^GSPC regression – fuzzy LSTM variants")
    p.add_argument("--variants", nargs="*", default=None,
                   choices=cfg.VARIANTS,
                   help="Variants to run (default: all)")
    p.add_argument("--seeds", type=int, default=None,
                   help="Number of seeds (0..N-1). Default: 10")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    seeds = list(range(args.seeds)) if args.seeds is not None else cfg.SEEDS
    main(variants=args.variants, seeds=seeds, make_plots=not args.no_plots)
