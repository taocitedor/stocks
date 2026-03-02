# vlab_backtest_ORA_final.py
import pandas as pd
import numpy as np
from google.cloud import bigquery

# -------------------
# UTILITAIRES VECTORISÉS
# -------------------

def SMA(series, period):
    return series.rolling(period).mean()

def ATR(df, period):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def pivot_points(df, window=3):
    """
    Version GAS : détecte pivots hauts/bas
    """
    highs = df['High'].values
    lows = df['Low'].values
    pivot_highs, pivot_lows = [], []

    for i in range(window, len(df)-window):
        isH = all(highs[i] >= highs[i-window:i+window+1][j] for j in range(2*window+1) if j != window)
        isL = all(lows[i] <= lows[i-window:i+window+1][j] for j in range(2*window+1) if j != window)
        if isH:
            pivot_highs.append((i, highs[i]))
        if isL:
            pivot_lows.append((i, lows[i]))
    return pivot_highs, pivot_lows

def structure_HL(pivots_high, pivots_low):
    """
    Calcule HH/HL en prenant les 2 derniers pivots de chaque type
    """
    hh = hl = False
    if len(pivots_high) >= 2 and len(pivots_low) >= 2:
        last_highs = [v for i,v in pivots_high[-2:]]
        last_lows = [v for i,v in pivots_low[-2:]]
        hh = last_highs[-1] > last_highs[-2]
        hl = last_lows[-1] > last_lows[-2]
    return hh, hl

def sqz_exact(df, period=20):
    closes = df['Close'].tail(period)
    highs = df['High'].tail(period)
    lows = df['Low'].tail(period)
    sma = closes.mean()
    sd = closes.std()
    atr = (highs - lows).mean()
    return (sma + 2*sd < sma + 1.5*atr) and (sma - 2*sd > sma - 1.5*atr)

def get_score(sub_df, index_df, logs=None):
    """
    Calcul du score identique au GAS pour ouverture position
    """
    if len(sub_df) < 63:
        if logs: logs.append(f"{sub_df['Date'].iloc[-1]} - Score skipped (not enough data)")
        return 0
    idx_slice = index_df[index_df['Date'] <= sub_df['Date'].iloc[-1]].tail(63)
    if len(idx_slice) < 63:
        if logs: logs.append(f"{sub_df['Date'].iloc[-1]} - Score skipped (not enough index data)")
        return 0

    cls = sub_df['Close']
    rsi = RSI(cls).iloc[-1]
    mm20 = cls.tail(20).mean()
    v_ratio = sub_df['Volume'].iloc[-1] / (sub_df['Volume'].tail(20).mean() or 1)
    pivots_high, pivots_low = pivot_points(sub_df, window=3)
    hh, hl = structure_HL(pivots_high, pivots_low)
    s_sqz = sqz_exact(sub_df)
    
    # --- Score détaillé identique GAS ---
    score = 0
    # RSI
    if 50 <= rsi <= 70: score += 15
    # Volume
    if v_ratio > 1.5: score += 25
    elif v_ratio > 1.1: score += 12.5
    # MM20
    close_vs_mm20 = abs(sub_df['Close'].iloc[-1] - mm20)/mm20
    if close_vs_mm20 <= 0.01:
        if sub_df['Close'].iloc[-1] >= mm20: score += 20
        else: score -= 10
    # Structure HH/HL
    if hh and hl: score += 30
    # Squeeze
    if s_sqz: score += 10

    if logs:
        logs.append(f"{sub_df['Date'].iloc[-1]} - RSI:{rsi:.2f}, v_ratio:{v_ratio:.2f}, MM20:{mm20:.2f}, HH:{hh}, HL:{hl}, Sqz:{s_sqz}, Score={score:.2f}")
    return score

# -------------------
# BACKTEST PRINCIPAL
# -------------------

def run_backtest_ORA():
    params = {
        'PROJECT_ID': 'project-16c606d0-6527-4644-907',
        'DATASET_ID': 'Trading',
        'TABLE_HISTO': 'CC_Historique_Cours',
        'INDEX_TICKER_LABO': '^FCHI',
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
        'VLAB_FEES': 0.0056,
        'VLAB_POS_SIZE': 4000
    }

    logs = ["=== START BACKTEST ORA.PA ==="]

    # --- Connexion BigQuery ---
    client = bigquery.Client(project=params['PROJECT_ID'])
    query = f"""
        SELECT * FROM `{params['DATASET_ID']}.{params['TABLE_HISTO']}` 
        WHERE Ticker IN ('ORA.PA','^FCHI') 
        ORDER BY Date ASC
    """
    df = client.query(query).to_dataframe()

    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    logs.append(f"Données récupérées : {len(df)} lignes pour ORA.PA + ^FCHI")

    df_or = df[df['Ticker']=='ORA.PA'].sort_values('Date').reset_index(drop=True)
    df_idx = df[df['Ticker']==params['INDEX_TICKER_LABO']].sort_values('Date').reset_index(drop=True)

    if params['VLAB_USE_MARKET_FILTER']:
        df_idx['SMA100'] = SMA(df_idx['Close'], params['VLAB_MARKET_SMA_PERIOD'])
        df_idx['SMA100_PREV'] = df_idx['SMA100'].shift(5)
        df_idx['SLOPE'] = (df_idx['SMA100'] - df_idx['SMA100_PREV']) / df_idx['SMA100_PREV']

    trades = []
    in_pos = False
    ePrice = 0
    hasReachedBE = False

    for i in range(params['VLAB_GLOBAL_SAMPLES'], len(df_or)):
        curr = df_or.loc[i]
        curr_date = curr['Date']
        curr_close = curr['Close']
        sub_df = df_or.loc[:i]

        slope = 0.0
        if params['VLAB_USE_MARKET_FILTER']:
            idx_row = df_idx[df_idx['Date']==curr_date]
            if idx_row.empty:
                last_date_before = df_idx[df_idx['Date'] < curr_date]['Date'].max()
                next_date_after = df_idx[df_idx['Date'] > curr_date]['Date'].min()
                logs.append(f"{curr_date.date()} - skipped (index date missing) | last before: {last_date_before} | next after: {next_date_after}")
                continue

            fchi_close = idx_row['Close'].values[0]
            fchi_sma100 = idx_row['SMA100'].values[0]
            slope = idx_row['SLOPE'].values[0]

            logs.append(f"{curr_date.date()} - FCHI close={fchi_close:.2f}, SMA100={fchi_sma100:.2f}, slope={slope:.5f}")

            if pd.notna(fchi_sma100) and fchi_close < fchi_sma100:
                logs.append(f"{curr_date.date()} - skipped (FCHI below SMA100)")
                continue

        # TP dynamique
        currentTP = params['VLAB_TP_TREND'] if slope >= params['VLAB_TREND_THRESHOLD'] else params['VLAB_TP_RANGE']
        atr_val = ATR(sub_df, params['VLAB_ATR_PERIOD']).iloc[-1]
        vol_pct = atr_val / curr_close
        currentBE = params['VLAB_BE_FAST'] if slope >= 0.004 and vol_pct < params['VLAB_VOLAT_LIMIT'] else params['VLAB_BE_SLOW']

        # --- Gestion position ouverte ---
        if in_pos:
            high_perf = (curr['High'] - ePrice)/ePrice
            low_perf = (curr['Low'] - ePrice)/ePrice
            if not hasReachedBE and high_perf >= currentBE:
                hasReachedBE = True
            effectiveSL = params['VLAB_FEES'] if hasReachedBE else -params['VLAB_BE_SLOW']
            hitTP = high_perf >= currentTP
            hitSL = low_perf <= effectiveSL
            if hitTP or hitSL:
                raw_exit = effectiveSL
                if hitTP and not hitSL:
                    raw_exit = currentTP
                trade_cash = (raw_exit - params['VLAB_FEES'])*params['VLAB_POS_SIZE']
                status = "GAGNÉ" if hitTP else "NEUTRE" if hasReachedBE else "PERDU"
                trades.append({
                    'Date': curr_date,
                    'Ticker': 'ORA.PA',
                    'EntryPrice': ePrice,
                    'ExitPrice': curr_close,
                    'Cash': trade_cash,
                    'Status': status
                })
                logs.append(f"Fermeture position ORA.PA le {curr_date.date()} Status={status} Cash={trade_cash:.2f}")
                in_pos = False
            continue

        # --- Ouverture position ---
        score = get_score(sub_df, df_idx, logs)
        if score >= params['VLAB_GLOBAL_SCORE']:
            in_pos = True
            ePrice = curr_close
            hasReachedBE = False
            logs.append(f"Ouverture position ORA.PA le {curr_date.date()} Close={curr_close:.2f} Score={score:.2f}")

    nb_total = len(trades)
    nb_gagnes = sum(1 for t in trades if t['Status']=="GAGNÉ")
    nb_neutres = sum(1 for t in trades if t['Status']=="NEUTRE")
    nb_perdus = sum(1 for t in trades if t['Status']=="PERDU")

    logs.append(f"\n=== RÉSULTATS GLOBAUX ===\nNombre de trades : {nb_total}\nGAGNÉS : {nb_gagnes}\nPERDUS : {nb_perdus}")

    return {
        'logs': logs,
        'nb_trades': nb_total,
        'nb_gagnes': nb_gagnes,
        'nb_neutres': nb_neutres,
        'nb_perdus': nb_perdus,
        'status': 'ok',
        'trades': trades
    }
