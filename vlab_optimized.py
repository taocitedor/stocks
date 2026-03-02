# vlab_optimized.py
from google.cloud import bigquery
import pandas as pd
import numpy as np

# -----------------------------
# FONCTIONS INDICATEURS VECTORISÉS
# -----------------------------

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
    return 100 - 100 / (1 + rs)

def pivot_points_vectorized(df, window=3):
    highs = df['High']
    lows = df['Low']
    piv_high = (highs == highs.rolling(window*2+1, center=True).max())
    piv_low = (lows == lows.rolling(window*2+1, center=True).min())
    return piv_high, piv_low

def structure_vectorized(piv_high, piv_low):
    hh = hl = pd.Series(False, index=piv_high.index)
    hh_values = piv_high.astype(int).rolling(2).apply(lambda x: x.iloc[-1] > x.iloc[-2] if len(x)==2 else False, raw=False)
    hl_values = piv_low.astype(int).rolling(2).apply(lambda x: x.iloc[-1] > x.iloc[-2] if len(x)==2 else False, raw=False)
    hh.loc[hh_values.index] = hh_values.fillna(False)
    hl.loc[hl_values.index] = hl_values.fillna(False)
    return hh, hl

def sqz_vectorized(df, period=20):
    closes = df['Close'].rolling(period)
    highs = df['High'].rolling(period)
    lows = df['Low'].rolling(period)
    sma = closes.mean()
    sd = closes.std()
    atr = (highs.max() - lows.min())
    return (sma + 2*sd < sma + 1.5*atr) & (sma - 2*sd > sma - 1.5*atr)

def get_score_vectorized(df, index_df, VLAB_GLOBAL_SCORE):
    rsi = RSI(df['Close'])
    mm20 = df['Close'].rolling(20).mean()
    vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean()
    piv_high, piv_low = pivot_points_vectorized(df)
    hh, hl = structure_vectorized(piv_high, piv_low)
    sqz = sqz_vectorized(df)
    
    score = pd.Series(0, index=df.index)
    score += np.where((rsi>=50) & (rsi<=70), 15, 0)
    score += np.where(vol_ratio>1.5, 25, np.where(vol_ratio>1.1, 12.5, 0))
    score += np.where((abs(df['Close']-mm20)/mm20 <= 0.01) & (df['Close']>=mm20), 20, 
                      np.where(abs(df['Close']-mm20)/mm20 <= 0.01, -10, 0))
    score += np.where(hh & hl, 30, 0)
    score += np.where(sqz, 10, 0)
    score = score.fillna(0)
    return score

# -----------------------------
# FONCTION BACKTEST ULTRA OPTIMISÉ
# -----------------------------

def run_vlab_optimized(params):
    """
    params = {
        'PROJECT_ID': 'xxx',
        'DATASET_ID': 'xxx',
        'TABLE_HISTO': 'xxx',
        'VLAB_GLOBAL_SCORE': 86,
        'VLAB_GLOBAL_SL': 0.10,
        'VLAB_POS_SIZE': 4000,
        'VLAB_FEES': 0.0056,
        'VLAB_BE_FAST': 0.06,
        'VLAB_BE_SLOW': 0.0495,
        'VLAB_ATR_PERIOD': 50,
        'VLAB_VOLAT_LIMIT': 0.025,
        'VLAB_TP_TREND': 0.13,
        'VLAB_TP_RANGE': 0.10,
        'VLAB_TREND_THRESHOLD': -0.003,
        'VLAB_MARKET_SMA_PERIOD': 100,
        'INDEX_TICKER_LABO': '^FCHI',
        'VLAB_USE_MARKET_FILTER': True
    }
    """
    client = bigquery.Client(project=params['PROJECT_ID'])
    query = f"SELECT * FROM `{params['PROJECT_ID']}.{params['DATASET_ID']}.{params['TABLE_HISTO']}` ORDER BY Date ASC"
    df = client.query(query).to_dataframe()
    df[['Close','High','Low','Volume']] = df[['Close','High','Low','Volume']].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    
    tickers = df['Ticker'].unique()
    index_df = df[df['Ticker']==params['INDEX_TICKER_LABO']].copy()
    index_df = index_df.sort_values('Date').reset_index(drop=True)
    index_df['SMA100'] = SMA(index_df['Close'], params['VLAB_MARKET_SMA_PERIOD'])
    index_df['SMA100_PREV'] = index_df['SMA100'].shift(5)
    index_df['SLOPE'] = (index_df['SMA100'] - index_df['SMA100_PREV']) / index_df['SMA100_PREV']
    
    nb_total=0
    nb_gagnes=0
    nb_neutres=0
    nb_perdus=0
    
    for t in tickers:
        if t == params['INDEX_TICKER_LABO']: continue
        df_t = df[df['Ticker']==t].copy().sort_values('Date').reset_index(drop=True)
        df_t['ATR'] = ATR(df_t, params['VLAB_ATR_PERIOD'])
        df_t['Score'] = get_score_vectorized(df_t, index_df, params['VLAB_GLOBAL_SCORE'])
        in_pos = False
        ePrice = 0
        hasBE = False
        
        # Joindre info SMA / slope
        df_t = df_t.merge(index_df[['Date','SMA100','SLOPE']], on='Date', how='left')
        
        for i,row in df_t.iterrows():
            if in_pos:
                high_perf = (row['High']-ePrice)/ePrice
                low_perf = (row['Low']-ePrice)/ePrice
                if not hasBE and high_perf>= (params['VLAB_BE_FAST'] if row['SLOPE']>=0.004 and (row['ATR']/row['Close'])<params['VLAB_VOLAT_LIMIT'] else params['VLAB_BE_SLOW']):
                    hasBE = True
                effectiveSL = params['VLAB_FEES'] if hasBE else -params['VLAB_GLOBAL_SL']
                hitTP = high_perf >= (params['VLAB_TP_TREND'] if row['SLOPE']>=params['VLAB_TREND_THRESHOLD'] else params['VLAB_TP_RANGE'])
                hitSL = low_perf <= effectiveSL
                if hitTP or hitSL:
                    if hitTP and not hitSL: raw_exit = params['VLAB_TP_TREND']
                    else: raw_exit = effectiveSL
                    trade_cash = (raw_exit - params['VLAB_FEES'])*params['VLAB_POS_SIZE']
                    status = "GAGNÉ" if hitTP else "NEUTRE" if hasBE else "PERDU"
                    nb_total +=1
                    if status=="GAGNÉ": nb_gagnes+=1
                    elif status=="NEUTRE": nb_neutres+=1
                    else: nb_perdus+=1
                    in_pos=False
                continue
            # ouverture position
            slope_ok = row['SLOPE']>=params['VLAB_TREND_THRESHOLD'] if params['VLAB_USE_MARKET_FILTER'] else True
            sma_ok = row['Close']>=row['SMA100'] if params['VLAB_USE_MARKET_FILTER'] else True
            if row['Score']>=params['VLAB_GLOBAL_SCORE'] and slope_ok and sma_ok:
                in_pos=True
                ePrice=row['Close']
                hasBE=False
    
    return {
        'TRADE_TOTAL': nb_total,
        'GAGNÉS': nb_gagnes,
        'NEUTRES': nb_neutres,
        'PERDUS': nb_perdus
    }
