import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #16213e);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid #0f3460;
        margin-bottom: 0.5rem;
    }
    .metric-label { color: #a0aec0; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em; }
    .metric-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; }
    .metric-delta-pos { color: #48bb78; font-size: 0.85rem; }
    .metric-delta-neg { color: #fc8181; font-size: 0.85rem; }
    .live-badge {
        display: inline-block; background: #c53030; color: white;
        border-radius: 20px; padding: 2px 10px; font-size: 0.7rem;
        font-weight: 700; letter-spacing: 0.1em; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #90cdf4;
        border-bottom: 2px solid #2d3748; padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STOCKS = {
    "Oracle": "ORCL",
    "ICICI Bank": "ICICIBANK.NS",
    "Britannia": "BRITANNIA.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Abbott India": "ABBOTINDIA.NS",
}

WINDOW_SIZE = 60
TRAIN_SPLIT = 0.80   # 80% train, 20% test


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)

    return df

@st.cache_data(ttl=60, show_spinner=False)
def get_live_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period="2d")
        return float(data["Close"].iloc[-1])
    except Exception:
        return np.nan


# ── Feature Engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA_30"] = df["Close"].rolling(30).mean()
    df["Volatility"] = df["Close"].rolling(30).std()
    df = df.dropna()
    return df


# ── Sequence Builder ──────────────────────────────────────────────────────────
def create_sequences(data: np.ndarray, window: int = 60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ── Model Training (LSTM-style via numpy – no TF dependency for Streamlit) ────
def run_simple_prediction(df: pd.DataFrame):
    """
    Uses a rolling-window linear regression as a lightweight predictor
    so the app runs without GPU / TensorFlow in a pure Streamlit env.
    For the real LSTM / GRU results, see the accompanying notebook.
    """
    from sklearn.linear_model import Ridge

    feat_df = add_features(df)
    data = feat_df[["Close", "MA_30", "Volatility"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    split = int(len(scaled) * TRAIN_SPLIT)
    train, test = scaled[:split], scaled[split:]

    X_train, y_train = create_sequences(train, WINDOW_SIZE)
    X_test, y_test = create_sequences(test, WINDOW_SIZE)

    # Flatten for Ridge
    X_tr = X_train.reshape(len(X_train), -1)
    X_te = X_test.reshape(len(X_test), -1)

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_train)

    preds_scaled = model.predict(X_te)

    # Inverse transform (only Close column matters)
    dummy = np.zeros((len(preds_scaled), 3))
    dummy[:, 0] = preds_scaled
    pred_prices = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((len(y_test), 3))
    dummy2[:, 0] = y_test
    actual_prices = scaler.inverse_transform(dummy2)[:, 0]

    test_dates = feat_df.index[split + WINDOW_SIZE:]

    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae = mean_absolute_error(actual_prices, pred_prices)
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100

    return test_dates, actual_prices, pred_prices, rmse, mae, mape
    
def predict_future_prices(df: pd.DataFrame, days: int = 5):
    from sklearn.linear_model import Ridge

    feat_df = add_features(df)

    data = feat_df[["Close", "MA_30", "Volatility"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = create_sequences(scaled, WINDOW_SIZE)

    X = X.reshape(len(X), -1)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Last sequence
    last_seq = scaled[-WINDOW_SIZE:]

    future_prices = []

    current_seq = last_seq.copy()

    for _ in range(days):
        x_input = current_seq.reshape(1, -1)

        pred_scaled = model.predict(x_input)[0]

        dummy = np.zeros((1, 3))
        dummy[0, 0] = pred_scaled

        pred_price = scaler.inverse_transform(dummy)[0, 0]

        future_prices.append(pred_price)

        # Create next row
        next_row = current_seq[-1].copy()
        next_row[0] = pred_scaled

        # Roll window
        current_seq = np.vstack([current_seq[1:], next_row])

    return future_prices

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    selected_name = st.selectbox("Select Stock", list(STOCKS.keys()))
    period = st.selectbox("History Period", ["1y", "2y", "3y", "5y"], index=2)
    auto_refresh = st.toggle("Auto-Refresh (60s)", value=False)
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This dashboard fetches live stock data via **yfinance**, "
        "trains a rolling-window model, and compares **predicted vs actual** "
        "close prices in real time.\n\n"
        "Deep-learning models (LSTM, GRU, ARIMA) are detailed in the "
        "accompanying Jupyter notebook."
    )
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

ticker = STOCKS[selected_name]

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("📈 Stock Price Prediction Dashboard")
st.markdown(
    f"**{selected_name}** (`{ticker}`) &nbsp;|&nbsp; "
    f"<span class='live-badge'>LIVE</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {selected_name}…"):
    df = load_stock_data(ticker, period)
    live_price = get_live_price(ticker)

if df.empty:
    st.error("Could not load data. Check your internet connection or try another stock.")
    st.stop()

# Run prediction
with st.spinner("Running prediction model…"):
    test_dates, actual_prices, pred_prices, rmse, mae, mape = run_simple_prediction(df)
    future_prices = predict_future_prices(df, days=5)
    future_dates = pd.bdate_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=5
)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

last_actual = actual_prices[-1]
last_pred = pred_prices[-1]

error_pct = abs(last_actual - last_pred) / last_actual * 100

# Safe close-price extraction
close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()

if len(close_series) > 1:
    prev_close = float(close_series.iloc[-2])
else:
    prev_close = float(last_actual)

# Live % change
if not np.isnan(live_price) and prev_close != 0:
    live_chg = ((live_price - prev_close) / prev_close) * 100
else:
    live_chg = 0
    
with k1:
    st.metric("Live Price", f"₹{live_price:,.2f}" if ".NS" in ticker else f"${live_price:,.2f}",
              f"{live_chg:+.2f}%")
with k2:
    st.metric("Last Predicted", f"₹{last_pred:,.2f}" if ".NS" in ticker else f"${last_pred:,.2f}")
with k3:
    st.metric("Prediction Error", f"{error_pct:.2f}%")
with k4:
    st.metric("RMSE", f"{rmse:,.2f}")
with k5:
    st.metric("MAPE", f"{mape:.2f}%")

st.markdown("---")

# ── Main Chart ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predicted vs Actual",
    "📉 Full History & Indicators",
    "📋 Metrics Table",
    "🔮 Future Forecast"
])

with tab1:
    st.markdown('<div class="section-header">Predicted vs Actual Close Price (Test Period)</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates, y=actual_prices,
        name="Actual Price",
        line=dict(color="#63b3ed", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Actual: <b>%{y:,.2f}</b><extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_prices,
        name="Predicted Price",
        line=dict(color="#f6ad55", width=2, dash="dot"),
        hovertemplate="%{x|%b %d, %Y}<br>Predicted: <b>%{y:,.2f}</b><extra></extra>"
    ))

    # Error band
    fig.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates[::-1]),
        y=list(pred_prices * 1.05) + list(pred_prices[::-1] * 0.95),
        fill="toself", fillcolor="rgba(246,173,85,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, name="Confidence Band"
    ))

    # Live price line
    if not np.isnan(live_price):
        fig.add_hline(y=live_price, line_dash="dash", line_color="#68d391",
                      annotation_text=f"Live: {live_price:,.2f}", annotation_position="bottom right")

    fig.update_layout(
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date", yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual plot
    residuals = actual_prices - pred_prices
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=test_dates, y=residuals, name="Residual",
                          marker_color=np.where(residuals >= 0, "#48bb78", "#fc8181")))
    fig2.update_layout(template="plotly_dark", height=200,
                       yaxis_title="Residual", xaxis_title="",
                       showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.markdown('<div class="section-header">Prediction Residuals</div>', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)


with tab2:
    feat_df = add_features(df)
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3],
                         subplot_titles=["Close Price + 30-Day MA", "30-Day Volatility"])

    fig3.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Close"],
                              name="Close", line=dict(color="#63b3ed", width=1.5)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=feat_df.index, y=feat_df["MA_30"],
                              name="MA 30", line=dict(color="#f6ad55", width=1.5, dash="dot")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Volatility"],
                              name="Volatility", line=dict(color="#fc8181", width=1.5),
                              fill="tozeroy", fillcolor="rgba(252,129,129,0.1)"), row=2, col=1)

    fig3.update_layout(template="plotly_dark", height=550, showlegend=True,
                       legend=dict(orientation="h", y=1.05),
                       margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True)


with tab3:
    st.markdown('<div class="section-header">Day-by-Day Comparison (Last 30 Days)</div>',
                unsafe_allow_html=True)

    n = min(30, len(test_dates))
    compare_df = pd.DataFrame({
        "Date": test_dates[-n:].strftime("%Y-%m-%d"),
        "Actual": np.round(actual_prices[-n:], 2),
        "Predicted": np.round(pred_prices[-n:], 2),
        "Error": np.round(actual_prices[-n:] - pred_prices[-n:], 2),
        "Error %": np.round(np.abs(actual_prices[-n:] - pred_prices[-n:]) / actual_prices[-n:] * 100, 2),
    }).reset_index(drop=True)

    def color_error(val):
        color = "#48bb78" if abs(val) < 2 else ("#f6ad55" if abs(val) < 5 else "#fc8181")
        return f"color: {color}"

    st.dataframe(compare_df, use_container_width=True, hide_index=True)
    # Download button
    csv = compare_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "comparison.csv", "text/csv")
with tab4:
    st.markdown(
        '<div class="section-header">5-Day Future Price Forecast</div>',
        unsafe_allow_html=True
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Predicted Price": np.round(future_prices, 2)
    })

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # Forecast chart
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#68d391", width=3),
        marker=dict(size=8)
    ))

    fig4.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Future Date",
        yaxis_title="Predicted Price",
        margin=dict(l=0, r=0, t=20, b=0)
    )

    st.plotly_chart(fig4, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sourced from Yahoo Finance via yfinance · "
    "Prediction uses ridge-regression on 60-day rolling windows · "
    "Deep-learning models (LSTM / GRU / ARIMA) are in Stock.ipynb"
)
