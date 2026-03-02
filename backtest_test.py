import pandas as pd
import numpy as np
from google.cloud import bigquery
import json

# -------------------
# CALCULS STANDARDS (FIDÉLITÉ 100% GOLD GAS)
# -------------------

def get_gold_pivots(df_slice, window=3):
    """Trouve les pivots High/Low comme GOLD_FIND_PIVOTS"""
    highs = df_slice['High'].values
    lows = df_slice['Low'].values
    p_h, p_l = [], []
    for i in range(window, len(df_slice) - window):
        # Logique stricte GAS : supérieur/inférieur aux voisins
        if all(highs[i] > highs[j] for j in range(i-window, i+window+1) if i != j): p_h.append(highs[i])
        if all(lows[i] < lows[j] for j in range(i-window, i+window+1) if i != j): p_l.append(lows[i])
    return p_h, p_l

def get_full_vlab_score(sub_df, df_idx):
    """Miroir exact de GOLD_CALCULATE_SCORE"""
    curr_idx = len(sub_df) - 1
    cls = sub_df['Close']
    
    # 1. VERROU FORCE RELATIVE (L'élément manquant qui bloquait ORA)
    # Comparaison Perf Stock vs Perf Index sur 63 bougies (J vs J-62)
    stock_perf = (cls.iloc[-1] - cls.iloc[-63]) / cls.iloc[-63]
    
    # On récupère l'index à la même date
    curr_date = sub_df['Date'].iloc[-1]
    idx_now = df_idx[df_idx['Date'] <= curr_date].iloc[-1]
    idx_past = df_idx[df_idx['Date'] <= sub_df['Date'].iloc[-63]].iloc[-1]
    index_perf = (idx_now['Close'] - idx_past['Close']) / idx_past['Close']
    
    rs_val = (stock_perf - index_perf) * 100
    if rs_val <= 0: return 0.0, "RS_NEG" # Verrou GAS

    # 2. RSI GOLD (Somme simple, pas de Wilder/EWM)
    diff = cls.tail(15).diff().dropna()
    gains = diff[diff > 0].sum()
    losses = -diff[diff < 0].sum()
    rsi = 100 if losses == 0 else 100 - (100 / (1 + (gains / losses)))

    # 3. VOLUME & MM20
    vol_mean_20 = sub_df['Volume'].tail(20).mean()
    v_ratio = sub_df['Volume'].iloc[-1] / (vol_mean_20 or 1)
    mm20 = cls.tail(20).mean()
    dist_mm20 = abs(cls.iloc[-1] - mm20) / mm20
    
    # 4. STRUCTURE (15 derniers pivots)
    p_h, p_l = get_gold_pivots(sub_df, window=3)
    p_h, p_l = p_h[-15:], p_l[-15:]
    hh = len(p_h) >= 2 and p_h[-1] > p_h[-2]
    hl = len(p_l) >= 2 and p_l[-1] > p_l[-2]
    
    # 5. SQUEEZE (Std Dev Population)
    std20 = np.std(cls.tail(20)) # np.std est par défaut en population (N) comme GAS
    atr20_sqz = (sub_df['High'].tail(20) - sub_df['Low'].tail(20)).mean()
    is_sqz = (mm20 + 2*std20 < mm20 + 1.5*atr20_sqz) and (mm20 - 2*std20 > mm20 - 1.5*atr20_sqz)

    # COMPTAGE DES POINTS
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
    df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.tz_localize(None)

    df_ora = df_raw[df_raw['Ticker']=='ORA.PA'].sort_values('Date').reset_index(drop=True)
    df_idx = df_raw[df_raw['Ticker']==p['INDEX_TICKER']].sort_values('Date').reset_index(drop=True)

    # Calculs Index
    df_idx['SMA'] = df_idx['Close'].rolling(p['VLAB_MARKET_SMA_PERIOD']).mean()
    df_idx['SLOPE'] = (df_idx['SMA'] - df_idx['SMA'].shift(4)) / df_idx['SMA'].shift(4)

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

        # Calcul Score (Passage de df_idx pour la Force Relative)
        score, reasons = get_full_vlab_score(df_ora.loc[:i], df_idx)
        
        if score >= p['VLAB_GLOBAL_SCORE']:
            # ATR pour Volatilité BE
            high_low = df_ora['High'] - df_ora['Low']
            high_close = (df_ora['High'] - df_ora['Close'].shift()).abs()
            low_close = (df_ora['Low'] - df_ora['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_val = tr.loc[:i].tail(p['VLAB_ATR_PERIOD']).mean()
            
            vol_pct = atr_val / curr['Close']
            active_tp = p['VLAB_TP_TREND'] if fchi_slope >= p['VLAB_TREND_THRESHOLD'] else p['VLAB_TP_RANGE']
            active_be_trig = p['VLAB_BE_FAST'] if (fchi_slope >= 0.004 and vol_pct < p['VLAB_VOLAT_LIMIT']) else p['VLAB_BE_SLOW']
            
            in_pos, ePrice, entry_date, reached_be = True, curr['Close'], curr['Date'], False
            debug_logs.append({"date": entry_date.strftime('%Y-%m-%d'), "score": score, "reasons": reasons})

    return {
        "gain_total": float(pd.DataFrame(trades)['Gain'].sum()) if trades else 0.0,
        "nb_trades": len(trades),
        "liste_trades": trades,
        "debug_buys": debug_logs
    }

if __name__ == "__main__":
    print(json.dumps(run_vlab_backtest_full(), indent=2))
