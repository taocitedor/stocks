# =========================
# backtest_test.py - VLAB v1.7.2 pour ORA.PA
# =========================
import pandas as pd
import numpy as np
import yfinance as yf

# -------------------------
# CONFIG
# -------------------------
TICKER = "ORA.PA"
INDEX_TICKER = "^FCHI"
START_DATE = "2020-01-01"
SCORE_THRESHOLD = 86
GLOBAL_SAMPLES = 63

# -------------------------
# INDICATEURS VLAB
# -------------------------
def vlab_rsi(close, period=14):
    gains, losses = 0, 0
    for i in range(len(close)-period, len(close)):
        delta = close.iloc[i] - close.iloc[i-1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def vlab_rs(stock_df, index_df):
    if len(stock_df) < 63:
        return 0
    p0s = stock_df['Close'].iloc[-1]
    p63s = stock_df['Close'].iloc[-63]
    current_date = stock_df.index[-1]
    
    idx_filtered = index_df[index_df.index <= current_date]
    if len(idx_filtered) < 63:
        return 0
    
    idx_sorted = idx_filtered.sort_index(ascending=False)
    idx0 = idx_sorted['Close'].iloc[0]
    idx63 = idx_sorted['Close'].iloc[62]
    
    return ((p0s - p63s)/p63s - (idx0 - idx63)/idx63) * 100

def vlab_pivots(df, w=3):
    pivots = []
    for i in range(w, len(df)-w):
        isH, isL = True, True
        for j in range(i-w, i+w+1):
            if j == i:
                continue
            if df['High'].iloc[j] >= df['High'].iloc[i]:
                isH = False
            if df['Low'].iloc[j] <= df['Low'].iloc[i]:
                isL = False
        if isH:
            pivots.append(("H", df['High'].iloc[i]))
        if isL:
            pivots.append(("L", df['Low'].iloc[i]))
    return pivots

def vlab_structure(pivots):
    highs = [p[1] for p in pivots if p[0] == "H"]
    lows  = [p[1] for p in pivots if p[0] == "L"]
    
    if len(highs) < 2 or len(lows) < 2:
        return "ND"
    
    isHH = highs[-1] > highs[-2]
    isHL = lows[-1] > lows[-2]
    
    return ("HH" if isHH else "LH") + "+" + ("HL" if isHL else "LL")

def vlab_squeeze(df):
    if len(df) < 20:
        return False
    last20 = df.tail(20)
    cls = last20['Close']
    sma = cls.mean()
    sd = cls.std()
    atr = (last20['High'] - last20['Low']).mean()
    return (sma + 2*sd < sma + 1.5*atr) and (sma - 2*sd > sma - 1.5*atr)

# -------------------------
# SCORING VLAB
# -------------------------
def vlab_score(stock_df, index_df, debug=False):
    score = 0
    rs = vlab_rs(stock_df, index_df)
    if rs <= 0:
        if debug: print("RS <= 0 → score = 0")
        return 0
    
    cls = stock_df['Close']
    rsi = vlab_rsi(cls)
    if 50 <= rsi <= 70:
        score += 15
    
    vol_mean = stock_df['Volume'].tail(20).mean()
    vol_ratio = stock_df['Volume'].iloc[-1] / (vol_mean if vol_mean != 0 else 1)
    
    if vol_ratio > 1.5:
        score += 25
    elif vol_ratio > 1.1:
        score += 12.5
    
    mm20 = cls.tail(20).mean()
    dist = abs(cls.iloc[-1] - mm20) / mm20
    if dist <= 0.01:
        score += 20 if cls.iloc[-1] >= mm20 else -10
    
    piv = vlab_pivots(stock_df)
    struct = vlab_structure(piv[-15:])
    if "HH" in struct and "HL" in struct:
        score += 30
    
    if vlab_squeeze(stock_df):
        score += 10
    
    if debug:
        print("RS:", round(rs,2))
        print("RSI:", round(rsi,2))
        print("Volume ratio:", round(vol_ratio,2))
        print("MM20 distance:", round(dist,4))
        print("Structure:", struct)
        print("Squeeze:", vlab_squeeze(stock_df))
        print("FINAL SCORE:", score)
    
    return score

# -------------------------
# FONCTION PRINCIPALE BACKTEST
# -------------------------
def run_backtest_ORA(debug_date=None):
    print("=== START BACKTEST ORA.PA ===")
    
    stock = yf.download(TICKER, start=START_DATE, progress=False)[['Close','High','Low','Volume']]
    index = yf.download(INDEX_TICKER, start=START_DATE, progress=False)[['Close','High','Low','Volume']]
    
    stock.dropna(inplace=True)
    index.dropna(inplace=True)
    
    if debug_date:
        stock = stock.loc[:debug_date]
    
    if len(stock) < GLOBAL_SAMPLES:
        print("Not enough data for backtest")
        return
    
    score = vlab_score(stock, index, debug=True)
    print("Score final:", score)
    print("Seuil entrée:", SCORE_THRESHOLD)
    print("Signal:", "ACHAT" if score >= SCORE_THRESHOLD else "PAS D'ENTRÉE")
    return score

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Debug sur une date historique pour vérifier signal
    run_backtest_ORA("2024-05-31")
