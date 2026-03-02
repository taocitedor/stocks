# vlab_optimized.py
from google.cloud import bigquery
import pandas as pd
import numpy as np

# -------------------
# FONCTIONS UTILITAIRES
# -------------------

def SMA(series, period):
    return series.rolling(period).mean()

def ATR(df, period):
    if len(df) < period:
        return pd.Series([0]*len(df))
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def RSI(series, period=14):
    if len(series) <= period:
        return pd.Series([50]*len(series))
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs.fillna(0)))

# Vectorisation pivots
def pivot_points(df, window=3):
    highs = df['High'].values
    lows = df['Low'].values
    pivots_high = np.zeros(len(df), dtype=bool)
    pivots_low = np.zeros(len(df), dtype=bool)
    for i in range(window, len(df)-window):
        if highs[i] == highs[i-window:i+window+1].max():
            pivots_high[i] = True
        if lows[i] == lows[i-window:i+window+1].min():
            pivots_low[i] = True
    return pivots_high, pivots_low

def structure_HL(pivots_high, pivots_low, highs, lows):
    hh = hl = False
    idx_h = np.where(pivots_high)[0]
    idx_l = np.where(pivots_low)[0]
    if len(idx_h) >= 2 and len(idx_l) >= 2:
        hh = highs[idx_h[-1]] > highs[idx_h[-2]]
        hl = lows[idx_l[-1]] > lows[idx_l[-2]]
    return hh, hl

def sqz_exact(df, period=20):
    if len(df) < period:
        return False
    closes = df['Close'].tail(period)
    highs = df['High'].tail(period)
    lows = df['Low'].tail(period)
    sma = closes.mean()
    sd = closes.std()
    atr = (highs - lows).mean()
    return (sma + 2*sd < sma + 1.5*atr) and (sma - 2*sd > sma - 1.5*atr)

def get_score(sub_df, index_df):
    if len(sub_df) < 63:
        return 0
    p0s = sub_df['Close'].iloc[-1]
    p63s = sub_df['Close'].iloc[-63]
    idx_slice = index_df[index_df['Date'] <= sub_df['Date'].iloc[-1]].tail(63)
    if len(idx_slice) < 63:
        return 0
    rs = ((p0s - p63s)/p63s - (idx_slice['Close'].iloc[-1]-idx_slice['Close'].iloc[0])/idx_slice['Close'].iloc[0])*100
    cls = sub_df['Close']
    rsi = RSI(cls).iloc[-1]
    mm20 = cls.tail(20).mean()
    v_ratio = sub_df['Volume'].iloc[-1] / (sub_df['Volume'].tail(20).mean() or 1)
    pivots_high, pivots_low = pivot_points(sub_df, window=3)
    hh, hl = structure_HL(pivots_high, pivots_low, sub_df['High'].values, sub_df['Low'].values)
    s_sqz = sqz_exact(sub_df)
    score = 0
    score += 15 if 50 <= rsi <= 70 else 0
    score += 25 if v_ratio > 1.5 else 12.5 if v_ratio > 1.1 else 0
    score += 20 if abs(p0s - mm20)/mm20 <= 0.01 and p0s >= mm20 else -10 if abs(p0s - mm20)/mm20 <= 0.01 else 0
    score += 30 if hh and hl else 0
    score += 10 if s_sqz else 0
    return score

# -------------------
# FONCTION PRINCIPALE
# -------------------

def run_backtest_ORA():
    """
    Backtest optimisé pour ORA.PA avec vectorisation
    """

    # --- Paramètres VLAB ---
    params = {
        'PROJECT_ID': 'project-16c606d0-6527-4644-907',
        'DATASET_ID': 'Trading',
        'TABLE_HISTO': 'CC_Historique_Cours',
        'INDEX_TICKER_LABO': 'FCHI',
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

    # --- Connexion BigQuery ---
    client = bigquery.Client(project=params['PROJECT_ID'])
    query = f"SELECT * FROM `{params['DATASET_ID']}.{params['TABLE_HISTO']}` WHERE Ticker='ORA.PA' OR Ticker='{params['INDEX_TICKER_LABO']}' ORDER BY Date ASC"
    df = client.query(query).to_dataframe()

    # Types
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])

    # Index
    index_df = df[df['Ticker']==params['INDEX_TICKER_LABO']].sort_values('Date').reset_index(drop=True)
    if params['VLAB_USE_MARKET_FILTER']:
        index_df['SMA100'] = SMA(index_df['Close'], params['VLAB_MARKET_SMA_PERIOD'])
        index_df['SMA100_PREV'] = index_df['SMA100'].shift(5)
        index_df['SLOPE'] = (index_df['SMA100'] - index_df['SMA100_PREV']) / index_df['SMA100_PREV']

    # Ticker ORA.PA uniquement
    df_t = df[df['Ticker']=='ORA.PA'].sort_values('Date').reset_index(drop=True)

    trades = []
    in_pos = False
    ePrice = 0
    hasReachedBE = False

    for i in range(params['VLAB_GLOBAL_SAMPLES'], len(df_t)):
        curr = df_t.iloc[i]
        sub_df = df_t.iloc[:i+1]

        # Filtre marché
        if params['VLAB_USE_MARKET_FILTER']:
            idx_row = index_df[index_df['Date'] == curr['Date']]
            if idx_row.empty or pd.isna(idx_row['SMA100'].iloc[0]) or curr['Close'] < idx_row['SMA100'].iloc[0]:
                continue
            slope = idx_row['SLOPE'].iloc[0]
        else:
            slope = 0.0

        # TP dynamique et BE
        currentTP = params['VLAB_TP_TREND'] if slope >= params['VLAB_TREND_THRESHOLD'] else params['VLAB_TP_RANGE']
        atr = ATR(sub_df, params['VLAB_ATR_PERIOD']).iloc[-1]
        vol_pct = atr / curr['Close']
        currentBE = params['VLAB_BE_FAST'] if slope >= 0.004 and vol_pct < params['VLAB_VOLAT_LIMIT'] else params['VLAB_BE_SLOW']

        # Trade ouvert
        if in_pos:
            high_perf = (curr['High'] - ePrice)/ePrice
            low_perf = (curr['Low'] - ePrice)/ePrice
            if not hasReachedBE and high_perf >= currentBE:
                hasReachedBE = True
            effectiveSL = params['VLAB_FEES'] if hasReachedBE else -params['VLAB_GLOBAL_SL']
            hitTP = high_perf >= currentTP
            hitSL = low_perf <= effectiveSL
            if hitTP or hitSL:
                if hitTP and not hitSL:
                    exit_price = ePrice*(1+currentTP)
                    status = "GAGNÉ"
                elif hitSL and hasReachedBE:
                    exit_price = ePrice*(1+effectiveSL)
                    status = "NEUTRE"
                else:
                    exit_price = ePrice*(1+effectiveSL)
                    status = "PERDU"
                trade_cash = (exit_price - ePrice - params['VLAB_FEES'])*params['VLAB_POS_SIZE']
                trades.append({
                    'Date': curr['Date'],
                    'Ticker': 'ORA.PA',
                    'EntryPrice': ePrice,
                    'ExitPrice': exit_price,
                    'Cash': trade_cash,
                    'Status': status
                })
                in_pos = False
            continue

        # Ouverture trade
        score = get_score(sub_df, index_df)
        if score >= params['VLAB_GLOBAL_SCORE']:
            in_pos = True
            ePrice = curr['Close']
            hasReachedBE = False

    # Stats
    nb_total = len(trades)
    nb_gagnes = sum(1 for t in trades if t['Status']=="GAGNÉ")
    nb_neutres = sum(1 for t in trades if t['Status']=="NEUTRE")
    nb_perdus = sum(1 for t in trades if t['Status']=="PERDU")

    # Logs détaillés
    logs = [f"Ouverture {t['Ticker']} le {t['Date'].date()} Entry={t['EntryPrice']} Exit={t['ExitPrice']} Cash={t['Cash']:.2f} Status={t['Status']}" for t in trades]

    return {
        'logs': logs,
        'nb_trades': nb_total,
        'nb_gagnes': nb_gagnes,
        'nb_neutres': nb_neutres,
        'nb_perdus': nb_perdus,
        'status': 'ok',
        'trades': trades
    }
