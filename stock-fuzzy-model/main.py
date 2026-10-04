"""Thin runner: full experiment via run_all.py (PLAN layout).

Run:
    python run_all.py            # all seeds x all variants -> results.csv
    python main.py               # same as run_all.py (convenience)
"""
from run_all import main

if __name__ == "__main__":
    main()
