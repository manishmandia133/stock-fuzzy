# Results

Ticker: ^GSPC · 2015-01-01..2025-12-31 · Regression (next-day Close via log-return) · Target: RMSE in price units

## Main table (mean ± std over seeds, test set)

| model                 |    RMSE | RMSE_std   |   nRMSE% |   MAPE% |      R |     MSE | MAE     | dir_acc   | vs_paper   | vs_persist   |   n_seeds |
|:----------------------|--------:|:-----------|---------:|--------:|-------:|--------:|:--------|:----------|:-----------|:-------------|----------:|
| Persistence           | 56.636  |            |   0.9711 |  0.6659 | 0.9949 | 3207.64 | 38.2959 | 0.000     | LOSS       | —            |         1 |
| Paper (Bhandari 2022) | 40.4574 |            |   1.2424 |  0.7989 | 0.9976 | 1637    | —       | —         | —          | LOSS         |         1 |
| Fuzzy_Input           | 56.4779 | 0.0539     |   0.9684 |  0.6639 | 0.9949 | 3189.76 | 38.1665 | 0.520     | LOSS       | WIN          |        10 |
| Fuzzy_Input_ensemble  | 56.4615 | —          |   0.9681 |  0.6635 | 0.9949 | 3187.91 | 38.1400 | 0.513     | LOSS       | WIN          |        10 |

WIN = beats reference on both RMSE and MAPE.  
Persistence: log-return = 0 (tomorrow = today).  
Paper: Bhandari et al. 2022 — ^GSPC 2006-2021.  
nRMSE% = RMSE / mean(Close_test) × 100; use this for cross-window comparisons.  
The paper test window averages ~$3,256, this test window ~$5,828.
