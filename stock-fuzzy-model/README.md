# Stock Fuzzy-LSTM Model

Converted from `fuzzy_lstm_100companies_v2 (1).ipynb`.

Hybrid market-direction predictor: Fuzzy Inference System + LSTM
(100 companies, US + India NSE).

## Structure

- `config.py` — ticker universe (60 US + 40 India), dates, window, hyper-params
- `data_loader.py` — batched `yfinance` download
- `features.py` — `return_1d`, `volatility_14d`, `target_direction`, scaling + sliding windows
- `fuzzy_controller.py` — volatility → learning-rate fuzzy system (`Low/Medium/High`)
- `model.py` — `DirectionLSTM`
- `train.py` — hybrid training loop (`use_fuzzy=True/False`), `evaluate()`
- `evaluation.py` — loss/LR plots + test directional accuracy benchmark
- `main.py` — end-to-end pipeline

## Run

```bash
pip install -r requirements.txt
python main.py
```
