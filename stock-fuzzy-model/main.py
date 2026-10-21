"""Thin runner: full experiment via run_all.py.

Run:
    python prepare_data.py   # cache data + write ^GSPC inspection CSV
    python main.py           # 30 seeds x 5 LR variants -> results.csv + figs/
"""
from run_all import main

if __name__ == "__main__":
    main()
