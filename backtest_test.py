import pandas as pd
import numpy as np
from google.cloud import bigquery
import json

# -------------------
# CALCULS STANDARDS (FIDÉLITÉ 100%)
# -------------------

def ATR(df, period):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def RSI_Wilder(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_pivots(df_slice, window=3):
    highs = df_slice['High'].values
    lows = df_slice['Low'].values
    p_h, p_l = [], []
    for i in range(window, len(df_slice) - window):
        if all(highs[i] >= highs[i-window:i+window+1]): p_h.append(highs[i])
        if all(lows[i] <= lows[i-window:i+window+1]): p_l.append(lows[i])
    return p_h, p_l

def get_full_vlab_score(sub_df):
    cls = sub_df['Close']
    rsi = RSI_Wilder(cls).iloc[-1]
    vol_prev_mean = sub_df['Volume'].iloc[:-1].tail(20).mean()
    v_ratio = sub_df['Volume'].iloc[-1] / (vol_prev_mean or 1)
    mm20 = cls.tail(20).mean()
    dist_mm20 = abs(cls.iloc[-1] - mm20) / mm20
    
    p_h, p_l = get_pivots(sub_df, window=3)
    hh = len(p_h) >= 2 and p_h[-1] >= p_h[-2]
    hl = len(p_l) >= 2 and p_l[-1] >= p_l[-2]
    
    std20 = cls.tail(20).std()
    atr20 = (sub_df['High'].tail(20) - sub_df['Low'].tail(20)).mean()
    is_sqz = (mm20 + 2*std20 < mm20 + 1.5*atr20) and (mm20 - 2*std20 > mm20 - 1.5*atr20)

    score = 0
    reasons = []
    if 50 <= rsi <= 70: 
        score += 15
        reasons.append("RSI")
    if v_ratio > 1.5: 
        score += 25
        reasons.append("VOL_B")
    elif v_ratio > 1.1: 
        score += 12.5
        reasons.append("VOL_M")
    if dist_mm20 <= 0.01:
        score += 20 if cls.iloc[-1] >= mm20 else -10
        reasons.append("MM20")
    if hh and hl: 
        score += 30
        reasons.append("STRUCT")
    if is_sqz: 
        score += 10
        reasons.append("SQZ")
    
    return float(score), "/".join(reasons)

# -------------------
# BACKTESTER GOLD STANDARD
# -------------------

def run_vlab_backtest_full():
    p = {
        'PROJECT_ID': 'project-16c606d0-6527-4644-907',
        'DATASET_ID': 'Trading',
        'TABLE_HISTO': 'CC_Historique_Cours',
        'INDEX_TICKER': '^FCHI',
        'VLAB_USE_MARKET_FILTER': True,
        'VLAB_MARKET_SMA_PERIOD': 100,
        'VLAB_GLOBAL_SCORE': 86,
        'VLAB_GLOBAL_SAMPLES': 63,
        'VLAB_TP_TREND': 0.13,
        'VLAB_TP_RANGE': 0.10,
        'VLAB_TREND_THRESHOLD': -0.003,
        'VLAB_ATR_PERIOD': 50,
        'VLAB_BE_FAST': 0.06,
        'VLAB_BE_SLOW': 0.0495,
        'VLAB_VOLAT_LIMIT': 0.025,
        'VLAB_SL': 0.10,
        'VLAB_FEES': 0.0056,
        'VLAB_POS_SIZE': 4000
    }

    client = bigquery.Client(project=p['PROJECT_ID'])
    query = f"SELECT * FROM `{p['DATASET_ID']}.{p['TABLE_HISTO']}` WHERE Ticker IN ('ORA.PA','{p['INDEX_TICKER']}') ORDER BY Date ASC"
    df_raw = client.query(query).to_dataframe()
    
    for col in ['Close','High','Low','Volume']: df_raw[col] = df_raw[col].astype(float)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    df_ora = df_raw[df_raw['Ticker']=='ORA.PA'].sort_values('Date').reset_index(drop=True)
    df_idx = df_raw[df_raw['Ticker']==p['INDEX_TICKER']].sort_values('Date').reset_index(drop=True)

    # Calculs Index
    df_idx['SMA'] = df_idx['Close'].rolling(p['VLAB_MARKET_SMA_PERIOD']).mean()
    df_idx['SLOPE'] = df_idx['SMA'].pct_change(5)

    trades = []
    debug_logs = []
    in_pos = False
    ePrice, entry_date, reached_be = 0, None, False
    active_tp, active_be_trig = 0, 0

    for i in range(p['VLAB_GLOBAL_SAMPLES'], len(df_ora)):
        curr = df_ora.loc[i]
        
        if in_pos:
            perf_h = (curr['High'] - ePrice) / ePrice
            perf_l = (curr['Low'] - ePrice) / ePrice
            if not reached_be and perf_h >= active_be_trig: reached_be = True
            
            # Gestion Sortie
            sl_price = p['VLAB_FEES'] if reached_be else -p['VLAB_SL']
            hit_tp = perf_h >= active_tp
            hit_sl = perf_l <= sl_price

            if hit_tp or hit_sl:
                final = active_tp if (hit_tp and not (hit_sl and perf_l < sl_price)) else sl_price
                trades.append({
                    "Achat": entry_date.strftime('%Y-%m-%d'),
                    "Vente": curr['Date'].strftime('%Y-%m-%d'),
                    "Gain": float((final - p['VLAB_FEES']) * p['VLAB_POS_SIZE']),
                    "BE_Reached": reached_be
                })
                in_pos = False
            continue

        # Filtre Marché SMA 100
        idx_row = df_idx[df_idx['Date'] == curr['Date']]
        if idx_row.empty: continue
        fchi_c, fchi_sma, fchi_slope = idx_row['Close'].iloc[0], idx_row['SMA'].iloc[0], idx_row['SLOPE'].iloc[0]

        if p['VLAB_USE_MARKET_FILTER'] and fchi_c < fchi_sma: continue

        # Calcul Score
        score, reasons = get_full_vlab_score(df_ora.loc[:i])
        
        if score >= p['VLAB_GLOBAL_SCORE']:
            # Calcul Volatilité pour BE
            atr_val = ATR(df_ora.loc[:i], p['VLAB_ATR_PERIOD']).iloc[-1]
            vol_pct = atr_val / curr['Close']
            
            # Application Paramètres Dynamiques
            active_tp = p['VLAB_TP_TREND'] if fchi_slope >= p['VLAB_TREND_THRESHOLD'] else p['VLAB_TP_RANGE']
            active_be_trig = p['VLAB_BE_FAST'] if (fchi_slope >= 0.004 and vol_pct < p['VLAB_VOLAT_LIMIT']) else p['VLAB_BE_SLOW']
            
            in_pos, ePrice, entry_date, reached_be = True, curr['Close'], curr['Date'], False
            debug_logs.append({
                "date": entry_date.strftime('%Y-%m-%d'),
                "score": score,
                "reasons": reasons,
                "tp_set": active_tp,
                "be_trig_set": active_be_trig
            })

    report_df = pd.DataFrame(trades)
    return {
        "gain_total": float(report_df['Gain'].sum()) if not report_df.empty else 0.0,
        "nb_trades": len(report_df),
        "liste_trades": trades,
        "debug_buys": debug_logs
    }

if __name__ == "__main__":
    print(json.dumps(run_backtest_ORA(), indent=2))
