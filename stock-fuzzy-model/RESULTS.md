# Results

Ticker: ^GSPC · 2015-01-01..2025-12-31 · Regression (next-day Close via log-return) · Target: RMSE in price units

## Main table (mean ± std over seeds, test set)

| model                 |    RMSE |   RMSE_std |   MAPE% |      R |     MSE | MAE     | dir_acc   | vs_paper   | vs_persist   |   n_seeds |
|:----------------------|--------:|-----------:|--------:|-------:|--------:|:--------|:----------|:-----------|:-------------|----------:|
| Persistence           | 56.636  |            |  0.6659 | 0.9949 | 3207.64 | 38.2959 | 0.000     | LOSS       | —            |         1 |
| Paper (Bhandari 2022) | 40.4574 |            |  0.7989 | 0.9976 | 1637    | —       | —         | —          | LOSS         |         1 |
| M2                    | 56.9472 |          0 |  0.6786 | 0.9949 | 3242.98 | 39.0853 | 0.462     | LOSS       | LOSS         |         1 |

WIN = beats reference on both RMSE and MAPE.  
Persistence: log-return = 0 (tomorrow = today).  
Paper: Bhandari et al. 2022 — ^GSPC 2006-2021.
