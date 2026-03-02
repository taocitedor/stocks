# vlab_optimized.py
from google.cloud import bigquery
import pandas as pd
import numpy as np

# -------------------
# FONCTIONS UTILITAIRES VECTORISÉES
# -------------------

def SMA(series, period):
    return series.rolling(period).mean()

def ATR(df, period):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def pivot_points(df, window=3):
    highs = df['High']
    lows = df['Low']
    pivot_highs, pivot_lows = [], []
    for i in range(window, len(df)-window):
        if highs[i] == highs[i-window:i+window+1].max():
            pivot_highs.append((i, highs[i]))
        if lows[i] == lows[i-window:i+window+1].min():
            pivot_lows.append((i, lows[i]))
    return pivot_highs, pivot_lows

def structure_HL(pivots_high, pivots_low):
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

def get_score(sub_df, index_df):
    if len(sub_df) < 63: return 0
    p0s = sub_df['Close'].iloc[-1]
    p63s = sub_df['Close'].iloc[-63]
    idx_slice = index_df[index_df['Date'] <= sub_df['Date'].iloc[-1]].tail(63)
    if len(idx_slice) < 63: return 0
    rs = ((p0s - p63s)/p63s - (idx_slice['Close'].iloc[-1]-idx_slice['Close'].iloc[0])/idx_slice['Close'].iloc[0])*100

    cls = sub_df['Close']
    rsi = RSI(cls).iloc[-1]
    mm20 = cls.tail(20).mean()
    v_ratio = sub_df['Volume'].iloc[-1] / (sub_df['Volume'].tail(20).mean() or 1)
    pivots_high, pivots_low = pivot_points(sub_df, window=3)
    hh, hl = structure_HL(pivots_high, pivots_low)
    s_sqz = sqz_exact(sub_df)
    
    score = 0
    score += 15 if 50 <= rsi <= 70 else 0
    score += 25 if v_ratio > 1.5 else 12.5 if v_ratio > 1.1 else 0
    score += 20 if abs(sub_df['Close'].iloc[-1] - mm20)/mm20 <= 0.01 and sub_df['Close'].iloc[-1]>=mm20 else -10 if abs(sub_df['Close'].iloc[-1] - mm20)/mm20 <= 0.01 else 0
    score += 30 if hh and hl else 0
    score += 10 if s_sqz else 0
    return score

# -------------------
# FONCTION PRINCIPALE : run_vlab
# -------------------

def run_vlab(params):
    """
    params: dict contenant tous les paramètres VLAB + info BigQuery
    """

    # --- Connexion BigQuery ---
    client = bigquery.Client(project=params['PROJECT_ID'])
    query = f"SELECT * FROM `{params['DATASET_ID']}.{params['TABLE_HISTO']}` ORDER BY Date ASC"
    df = client.query(query).to_dataframe()

    # Conversion types
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Tickeurs et dataframe de l'index
    tickers = df['Ticker'].unique()
    index_df = df[df['Ticker']==params['INDEX_TICKER_LABO']].sort_values('Date').reset_index(drop=True)
    if params['VLAB_USE_MARKET_FILTER']:
        index_df['SMA100'] = SMA(index_df['Close'], params['VLAB_MARKET_SMA_PERIOD'])
        index_df['SMA100_PREV'] = index_df['SMA100'].shift(5)
        index_df['SLOPE'] = (index_df['SMA100'] - index_df['SMA100_PREV']) / index_df['SMA100_PREV']
    
    # -------------------
    # Fonction interne simulation par ticker
    # -------------------
    def simulate_trades(ticker_df):
        trades = []
        in_pos = False
        ePrice = 0
        hasReachedBE = False
        ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)

        for i in range(params['VLAB_GLOBAL_SAMPLES'], len(ticker_df)):
            curr = ticker_df.loc[i]
            sub_df = ticker_df.loc[:i]
            curr_close = curr['Close']
            curr_date = curr['Date']

            # Filtre marché activable
            if params['VLAB_USE_MARKET_FILTER']:
                idx_row = index_df[index_df['Date']==curr_date]
                if idx_row.empty or curr_close < idx_row['SMA100'].values[0]:
                    continue
                slope = idx_row['SLOPE'].values[0]
            else:
                slope = 0.0

            # TP dynamique
            currentTP = params['VLAB_TP_TREND'] if slope >= params['VLAB_TREND_THRESHOLD'] else params['VLAB_TP_RANGE']
            atr = ATR(sub_df.tail(params['VLAB_ATR_PERIOD'])).iloc[-1]
            vol_pct = atr / curr_close
            currentBE = params['VLAB_BE_FAST'] if slope >= 0.004 and vol_pct < params['VLAB_VOLAT_LIMIT'] else params['VLAB_BE_SLOW']

            # -------------------
            # Position ouverte ?
            # -------------------
            if in_pos:
                high_perf = (curr['High'] - ePrice)/ePrice
                low_perf = (curr['Low'] - ePrice)/ePrice
                if not hasReachedBE and high_perf >= currentBE:
                    hasReachedBE = True
                effectiveSL = params['VLAB_FEES'] if hasReachedBE else -params['VLAB_GLOBAL_SL']
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
                        'Ticker': ticker_df['Ticker'].iloc[0],
                        'EntryPrice': ePrice,
                        'ExitPrice': curr_close,
                        'Cash': trade_cash,
                        'Status': status
                    })
                    in_pos = False
                continue

            # -------------------
            # Ouverture position
            # -------------------
            score = get_score(sub_df, index_df)
            if score >= params['VLAB_GLOBAL_SCORE']:
                in_pos = True
                ePrice = curr_close
                hasReachedBE = False

        return trades

    # -------------------
    # Boucle sur tous les tickers
    # -------------------
    all_trades_detail = []
    for t in tickers:
        if t == params['INDEX_TICKER_LABO']: 
            continue
        df_t = df[df['Ticker']==t].copy()
        trades = simulate_trades(df_t)
        all_trades_detail.extend(trades)

    # Stats globales
    nb_total = len(all_trades_detail)
    nb_gagnes = sum(1 for t in all_trades_detail if t['Status']=="GAGNÉ")
    nb_neutres = sum(1 for t in all_trades_detail if t['Status']=="NEUTRE")
    nb_perdus = sum(1 for t in all_trades_detail if t['Status']=="PERDU")

    return {
        'TRADE_TOTAL': nb_total,
        'GAGNÉS': nb_gagnes,
        'NEUTRES': nb_neutres,
        'PERDUS': nb_perdus,
        'TRADES_DETAIL': all_trades_detail
    }
