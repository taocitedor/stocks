import pandas as pd
import numpy as np
from google.cloud import bigquery

# -------------------
# UTILITAIRES DE CALCUL (STANDARD OR)
# -------------------

def SMA(series, period):
    return series.rolling(period).mean()

def ATR(df, period):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def RSI_Wilder(series, period=14):
    """RSI Wilder (Smoothing exponentiel) - Standard Labo"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_pivots_anti_bias(df_slice, window=3):
    """Détecte les pivots validés SANS regarder le futur (validation à i+window)"""
    highs = df_slice['High'].values
    lows = df_slice['Low'].values
    p_highs, p_lows = [], []
    # On s'arrête à len-window pour ne pas tricher
    for i in range(window, len(df_slice) - window):
        if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if i!=j):
            p_highs.append(highs[i])
        if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if i!=j):
            p_lows.append(lows[i])
    return p_highs, p_lows

def calculate_vlab_score(sub_df):
    """Calcul du score 0-100 identique au GAS"""
    cls = sub_df['Close']
    rsi = RSI_Wilder(cls).iloc[-1]
    # Volume Ratio vs 20 jours précédents (excluant J)
    vol_prev_mean = sub_df['Volume'].iloc[:-1].tail(20).mean()
    v_ratio = sub_df['Volume'].iloc[-1] / (vol_prev_mean or 1)
    
    mm20 = cls.tail(20).mean()
    dist_mm20 = abs(cls.iloc[-1] - mm20) / mm20
    
    p_h, p_l = get_pivots_anti_bias(sub_df, window=3)
    hh = len(p_h) >= 2 and p_h[-1] >= p_h[-2]
    hl = len(p_l) >= 2 and p_l[-1] >= p_l[-2]
    
    std20 = cls.tail(20).std()
    atr20 = (sub_df['High'].tail(20) - sub_df['Low'].tail(20)).mean()
    is_sqz = (mm20 + 2*std20 < mm20 + 1.5*atr20) and (mm20 - 2*std20 > mm20 - 1.5*atr20)

    score = 0
    if 50 <= rsi <= 70: score += 15
    if v_ratio > 1.5: score += 25
    elif v_ratio > 1.1: score += 12.5
    if dist_mm20 <= 0.01:
        score += 20 if cls.iloc[-1] >= mm20 else -10
    if hh and hl: score += 30
    if is_sqz: score += 10
    return score

# -------------------
# BACKTESTER V-LAB
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

    # Fetch Data
    client = bigquery.Client(project=p['PROJECT_ID'])
    query = f"SELECT * FROM `{p['DATASET_ID']}.{p['TABLE_HISTO']}` WHERE Ticker IN ('ORA.PA','{p['INDEX_TICKER']}') ORDER BY Date ASC"
    df_raw = client.query(query).to_dataframe()
    
    for col in ['Close', 'High', 'Low', 'Volume']: df_raw[col] = df_raw[col].astype(float)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    df_ora = df_raw[df_raw['Ticker']=='ORA.PA'].sort_values('Date').reset_index(drop=True)
    df_idx = df_raw[df_raw['Ticker']==p['INDEX_TICKER']].sort_values('Date').reset_index(drop=True)

    # Pré-calculs Index
    df_idx['SMA100'] = SMA(df_idx['Close'], p['VLAB_MARKET_SMA_PERIOD'])
    df_idx['SLOPE'] = df_idx['SMA100'].pct_change(5)

    trades = []
    in_pos = False
    ePrice = 0
    hasReachedBE = False
    active_tp = 0
    active_be_trigger = 0

    for i in range(p['VLAB_GLOBAL_SAMPLES'], len(df_ora)):
        curr = df_ora.loc[i]
        sub_df = df_ora.loc[:i]
        
        # 1. GESTION SORTIE (Prioritaire)
        if in_pos:
            perf_h = (curr['High'] - ePrice) / ePrice
            perf_l = (curr['Low'] - ePrice) / ePrice
            
            if not hasReachedBE and perf_h >= active_be_trigger:
                hasReachedBE = True
            
            # SL dynamique : Si BE activé -> Sortie aux frais, sinon SL 10%
            effective_sl = p['VLAB_FEES'] if hasReachedBE else -p['VLAB_SL']
            
            hit_tp = perf_h >= active_tp
            hit_sl = perf_l <= effective_sl

            if hit_tp or hit_sl:
                # Priorité SL en cas de bougie ambiguë
                final_perf = effective_sl if (hit_sl and not hit_tp) else active_tp
                trades.append({
                    'Date_Achat': entry_date,
                    'Date_Vente': curr['Date'],
                    'Status': "GAGNÉ" if final_perf > 0.05 else ("NEUTRE" if hasReachedBE else "PERDU"),
                    'Gain_Net': (final_perf - p['VLAB_FEES']) * p['VLAB_POS_SIZE']
                })
                in_pos = False
            continue

        # 2. FILTRE DE MARCHÉ (Application de l'interrupteur)
        idx_row = df_idx[df_idx['Date'] == curr['Date']]
        if idx_row.empty: continue
        
        fchi_c = idx_row['Close'].values[0]
        fchi_sma = idx_row['SMA100'].values[0]
        fchi_slope = idx_row['SLOPE'].values[0]

        if p['VLAB_USE_MARKET_FILTER'] and fchi_c < fchi_sma:
            continue

        # 3. SCORE ET ENTRÉE
        score = calculate_vlab_score(sub_df)
        if score >= p['VLAB_GLOBAL_SCORE']:
            atr_val = ATR(sub_df, p['VLAB_ATR_PERIOD']).iloc[-1]
            vol_pct = atr_val / curr['Close']
            
            # Calcul des cibles dynamiques au moment de l'achat
            active_tp = p['VLAB_TP_TREND'] if fchi_slope >= p['VLAB_TREND_THRESHOLD'] else p['VLAB_TP_RANGE']
            active_be_trigger = p['VLAB_BE_FAST'] if (fchi_slope >= 0.004 and vol_pct < p['VLAB_VOLAT_LIMIT']) else p['VLAB_BE_SLOW']
            
            in_pos = True
            ePrice = curr['Close']
            entry_date = curr['Date']
            hasReachedBE = False

    # Résultat Final
    report = pd.DataFrame(trades)
    print(f"\n=== BACKTEST V-LAB FINAL (Filter: {p['VLAB_USE_MARKET_FILTER']}) ===")
    print(f"Gain Total : {report['Gain_Net'].sum():.2f} €")
    print(f"Nb Trades  : {len(report)}")
    print(report['Status'].value_counts() if not report.empty else "Aucun trade")
    return report

if __name__ == "__main__":
    run_vlab_backtest_full()
