# frontend/app.py (v3 - 全チャート一覧表示・ショートカット付き)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
import json
import yfinance as yf
import plotly.graph_objects as go
from streamlit.components.v1 import html
import traceback
import pytz

# --- 設定 ---
API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = "/predict/v1/"
REFRESH_INTERVAL_SECONDS = 900
CHART_PERIODS = {"5m": "7d", "15m": "7d", "1h": "30d", "1d": "1y"}
BARS_TO_SHOW = 240
CHART_INTERVALS = ["5m", "15m", "1h", "1d"]
CURRENCY_PAIRS = [
    "USDJPY", "EURUSD", "EURJPY", "GBPUSD", "GBPJPY",
    "EURGBP", "AUDUSD", "EURAUD", "AUDJPY", "GBPAUD"
]
DISPLAY_NAMES = {pair: f"{pair[:3]}/{pair[3:]}" for pair in CURRENCY_PAIRS}
TICKER_MAP = {pair: f"{pair}=X" for pair in CURRENCY_PAIRS}

# --- ヘルパー関数 ---

@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS - 30)
def get_prediction_from_api(_cache_key, symbol: str):
    """FastAPIから予測結果を取得 (キャッシュ付き)"""
    api_url = f"{API_BASE_URL}{PREDICT_ENDPOINT}{symbol}"
    print(f"--- API Request ({time.strftime('%H:%M:%S')}) ---> {api_url}")
    log_messages.append(f"{time.strftime('%H:%M:%S')} - API Req: {symbol}")
    try:
        response = requests.get(api_url, timeout=180)
        response.raise_for_status()
        result = response.json()
        log_messages.append(f"{time.strftime('%H:%M:%S')} - API Res: {symbol} OK")
        print(f"--- API Response ({symbol}) OK ---")
        return result
    except requests.exceptions.Timeout:
        error_msg = f"API Timeout ({symbol})"; log_messages.append(f"{time.strftime('%H:%M:%S')} - {error_msg}"); st.error(error_msg); return {"symbol": symbol, "error_message": "Timeout"}
    except requests.exceptions.RequestException as e:
        error_msg = f"API Connect Err ({symbol}): {e}"; log_messages.append(f"{time.strftime('%H:%M:%S')} - {error_msg}"); st.error(error_msg); return {"symbol": symbol, "error_message": "Connection Error"}
    except Exception as e:
        error_msg = f"API Prediction Err ({symbol}): {e}"; log_messages.append(f"{time.strftime('%H:%M:%S')} - {error_msg}"); st.error(error_msg); traceback.print_exc(); return {"symbol": symbol, "error_message": "Prediction Error"}

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex を 1段 Index に変換"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:,~df.columns.duplicated()]
    if "Adj Close" in df.columns and "Close" in df.columns:
        df = df.drop("Adj Close", axis=1, errors='ignore')
    return df

# ★★★ チャートデータ取得関数 (キャッシュ付き) ★★★
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS - 10) # 更新間隔より少し短くキャッシュ
def fetch_chart_data(_cache_key, symbol, interval, period):
    """指定された通貨ペア・時間足のチャートデータを取得・整形"""
    print(f"  Fetching chart data: {symbol} ({interval}, {period})...")
    ticker = TICKER_MAP.get(symbol)
    if not ticker: return pd.DataFrame()
    try:
        df_chart = yf.download(ticker, interval=interval, period=period, progress=False, group_by="column")
        if not df_chart.empty:
            df_chart = _flatten_columns(df_chart)
            df_chart.columns = [str(c).capitalize() for c in df_chart.columns]
            if df_chart.index.tz is not None: df_chart.index = df_chart.index.tz_localize(None)
            keep_cols = ["Open", "High", "Low", "Close"]
            if all(col in df_chart.columns for col in keep_cols):
                df_chart = df_chart[keep_cols].copy()
                df_chart.dropna(inplace=True)
                # 最新 BARS_TO_SHOW 本に絞る
                if len(df_chart) >= BARS_TO_SHOW: df_chart = df_chart.tail(BARS_TO_SHOW)
                # print(f"   -> {interval}: {len(df_chart)} bars fetched.") # ログ簡略化
                return df_chart
            else: print(f"   -> OHLC missing for {symbol} {interval}"); return pd.DataFrame()
        else: return pd.DataFrame()
    except Exception as e: print(f"   -> Error fetching {symbol} {interval}: {e}"); return pd.DataFrame()

# チャート描画関数 (変更なし)
def create_candlestick_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    if not df.empty and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=title, increasing_line_color='red', decreasing_line_color='blue', increasing_line_width=1, decreasing_line_width=1, increasing_fillcolor='rgba(255,0,0,0.6)', decreasing_fillcolor='rgba(0,0,255,0.6)'))
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, xaxis_rangebreaks=[dict(bounds=["sat", "mon"])], margin=dict(l=30, r=30, t=50, b=30), height=300)
    else: fig.update_layout(title=f"{title} (データなし)", height=300)
    return fig

# --- Streamlit アプリ本体 ---
st.set_page_config(page_title="FX予測サマリー", layout="wide")
st.title("🤖 FX 10通貨ペア 予測サマリー")

# --- ログ表示エリア (サイドバー) ---
st.sidebar.header("処理ログ")
log_area = st.sidebar.empty()
log_messages = []
def update_log_display():
    log_area.text_area("ログ", "\n".join(log_messages), height=400, key=f"log_{time.time()}")

log_messages.append(f"{time.strftime('%H:%M:%S')} - ページ実行開始")
update_log_display()

# --- 全通貨ペアの予測を取得 & 表示 ---
cache_key = time.time() // (REFRESH_INTERVAL_SECONDS / 2) # キャッシュキー更新
all_predictions = {}
prediction_placeholders = {}

st.header("予測一覧")
status_message_area = st.empty()
status_message_area.write("各通貨ペアの予測を取得中...")

# テーブルヘッダー表示
N_COLS_TABLE = 8 # テーブルの列数
table_cols = st.columns(N_COLS_TABLE)
headers = ["通貨ペア", "5分足予測", "確率(U/F/D)", "現在価格", "日足予測", "時間足(2/4/6h)", "15分足(2/4h)", "更新時刻"]
for col, header in zip(table_cols, headers): col.write(f"**{header}**")

# 各行のプレースホルダ作成
for symbol in CURRENCY_PAIRS:
     cols = st.columns(N_COLS_TABLE)
     prediction_placeholders[symbol] = {"symbol": cols[0].empty(), "m5_pred": cols[1].empty(), "m5_prob": cols[2].empty(), "current_price": cols[3].empty(), "d1_pred": cols[4].empty(), "h1_pred": cols[5].empty(), "m15_pred": cols[6].empty(), "timestamp": cols[7].empty()}
     prediction_placeholders[symbol]["symbol"].write(DISPLAY_NAMES.get(symbol, symbol))
     for key in prediction_placeholders[symbol]:
         if key != 'symbol': prediction_placeholders[symbol][key].write("取得中...")

# API呼び出しと表示更新ループ
all_results_valid = True
for symbol in CURRENCY_PAIRS:
    prediction = get_prediction_from_api(cache_key, symbol) # キャッシュキーを渡す
    update_log_display() # ログ更新
    ph = prediction_placeholders[symbol]
    if prediction and prediction.get("predicted_class") is not None:
        all_predictions[symbol] = prediction # 念のため結果を保持
        pred_class = prediction['predicted_class']; prob_u=prediction.get('probability_up',0.0); prob_f=prediction.get('probability_flat',0.0); prob_d=prediction.get('probability_down',0.0)
        ts_str = pd.to_datetime(prediction.get('timestamp', '')).strftime('%H:%M:%S') if prediction.get('timestamp') else "--:--:--"
        m5_pred_text = f"➖FLAT ({prob_f:.1%})"
        if pred_class == "UP": m5_pred_text = f"🔼UP ({prob_u:.1%})"
        elif pred_class == "DOWN": m5_pred_text = f"🔽DOWN ({prob_d:.1%})"
        price_decimals = 2 if "JPY" in symbol.upper() else 4; fmt = f".{price_decimals}f"
        latest_close = prediction.get('latest_close'); d1_p = prediction.get('d1_pred_close'); h1_p2=prediction.get('h1_pred_2h'); h1_p4=prediction.get('h1_pred_4h'); h1_p6=prediction.get('h1_pred_6h'); m15_p8=prediction.get('m15_pred_8p'); m15_p16=prediction.get('m15_pred_16p')
        ph["m5_pred"].markdown(f"**{m5_pred_text}**"); ph["m5_prob"].progress(max(prob_u, prob_d, prob_f))
        ph["current_price"].text(f"{latest_close:{fmt}}" if latest_close else "N/A")
        ph["d1_pred"].text(f"{d1_p:{fmt}}" if d1_p else "N/A"); ph["h1_pred"].text(f"{h1_p2:{fmt}}/{h1_p4:{fmt}}/{h1_p6:{fmt}}" if h1_p2 else "N/A"); ph["m15_pred"].text(f"{m15_p8:{fmt}}/{m15_p16:{fmt}}" if m15_p8 else "N/A")
        ph["timestamp"].text(ts_str)
    else: error_msg=prediction.get("error_message","不明") if prediction else "失敗"; ph["m5_pred"].error(f"エラー:{error_msg}"); all_results_valid = False; # Handle error display

# 予測取得中のメッセージを消去
status_message_area.empty()

# --- チャートへのショートカット ---
st.divider()
st.header("チャートへのショートカット")
# 5列で表示する例
shortcut_cols = st.columns(5)
col_idx = 0
for symbol in CURRENCY_PAIRS:
    anchor = f"chart_{symbol.lower()}" # アンカー名を定義
    display_name = DISPLAY_NAMES.get(symbol, symbol)
    shortcut_cols[col_idx % 5].markdown(f"[{display_name}](#{anchor})") # Markdownリンク
    col_idx += 1

# --- 全通貨ペアのチャート表示 ---
st.divider()
st.header("全通貨ペア チャート")
log_messages.append(f"{time.strftime('%H:%M:%S')} - 全チャート描画開始")
update_log_display()

chart_cache_key = time.time() // (REFRESH_INTERVAL_SECONDS * 2) # チャートはAPIより長めにキャッシュ

for symbol in CURRENCY_PAIRS:
    # ↓↓↓ アンカーを設置 ↓↓↓
    st.markdown(f"<a name='chart_{symbol.lower()}'></a>", unsafe_allow_html=True)
    # エキスパンダーを使う場合:
    # with st.expander(f"{DISPLAY_NAMES.get(symbol, symbol)} のチャートを表示"):
    # エキスパンダーを使わない場合:
    st.subheader(f"{DISPLAY_NAMES.get(symbol, symbol)} チャート")
    chart_display_cols = st.columns(4) # 各通貨ペア内で4列
    chart_titles = ["5分足", "15分足", "1時間足", "日足"]

    for i, interval in enumerate(CHART_INTERVALS):
        period = CHART_PERIODS.get(interval)
        # ★★★ キャッシュ付き関数でチャートデータを取得 ★★★
        df_chart = fetch_chart_data(f"{symbol}_{interval}_{chart_cache_key}", symbol, interval, period) # キャッシュキーを追加

        with chart_display_cols[i]:
            if df_chart is not None and not df_chart.empty:
                 fig = create_candlestick_chart(df_chart, chart_titles[i])
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.warning(f"{chart_titles[i]} データ取得失敗")
    st.divider() # 各通貨ペアのチャートの後に区切り線

log_messages.append(f"{time.strftime('%H:%M:%S')} - 全チャート描画完了")
update_log_display()

# --- 自動更新 ---
st.sidebar.divider()
st.sidebar.caption(f"最終更新: {time.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.caption(f"{REFRESH_INTERVAL_SECONDS} 秒後に自動で再実行します...")
time.sleep(REFRESH_INTERVAL_SECONDS)
print(f"{time.strftime('%H:%M:%S')} - 自動再実行 (st.rerun)")
log_messages.append(f"{time.strftime('%H:%M:%S')} - 自動再実行開始")
update_log_display()
st.rerun()