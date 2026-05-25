# 📈 Stock Price Prediction Dashboard

A live Streamlit web application that fetches real-time stock data and compares **predicted vs actual** closing prices for 5 stocks — Oracle, ICICI Bank, Britannia, Maruti Suzuki, and Abbott India.

## 🚀 Live Demo

Deploy instantly to **Streamlit Community Cloud** (free):

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set `app.py` as entry point → Deploy

---

## 📦 Project Structure

```
stock_prediction_app/
├── app.py              ← Streamlit dashboard (main entry point)
├── Stock.ipynb         ← Original research notebook (LSTM / GRU / ARIMA)
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml     ← Dark-theme config
└── README.md
```

---

## 🏃 Run Locally

```bash
git clone https://github.com/<your-username>/stock_prediction_app.git
cd stock_prediction_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Models Used

| Location | Model | Purpose |
|---|---|---|
| `app.py` | Ridge Regression (60-day rolling window) | Lightweight live predictor |
| `Stock.ipynb` | LSTM (2-layer, 64→32 units) | Deep sequence model |
| `Stock.ipynb` | GRU (2-layer, 64→32 units) | Gated recurrent model |
| `Stock.ipynb` | ARIMA (auto-selected via pmdarima) | Classical time-series baseline |

---

## 📊 Features

- **Live price** fetched from Yahoo Finance every 60 seconds (toggle in sidebar)
- **Predicted vs Actual** chart with error band and residuals
- **Full history** chart with 30-day moving average and volatility
- **Day-by-day comparison table** with colour-coded error % (green < 2 %, amber < 5 %, red > 5 %)
- **CSV export** of the last 30-day comparison
- Supports 5 stocks across NSE (India) and NYSE (US)

---

## 🛠️ Tech Stack

- **Streamlit** — UI framework
- **yfinance** — Yahoo Finance data API
- **scikit-learn** — Ridge regression, MinMaxScaler, metrics
- **Plotly** — Interactive charts
- **TensorFlow / Keras** — LSTM & GRU (notebook only)
- **pmdarima** — Auto ARIMA (notebook only)

---

## 📄 License

MIT
