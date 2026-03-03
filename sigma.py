import pandas as pd
import numpy as np
from google.cloud import bigquery

# --- CONFIGURATION SYSTÈME ALPHA ---
ALPHA_CFG = {
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours',
    'IDX': '^FCHI',
    'MKT_FILTER': True,
    'SMA_P': 100,
    'MIN_SCORE': 86,
    'LOOKBACK': 63,
    'TP_TREND': 0.13,
    'TP_RANGE': 0.10,
    'SLOPE_TRESH': -0.003,
    'ATR_P': 50,
    'BE_F': 0.06,
    'BE_S': 0.0495,
    'VOL_LIM': 0.025,
    'STOP_L': 0.10,
    'FEES': 0.0056,
    'SIZE': 4000
}

def alpha_engine_v3():
    # 1. ACQUISITION DES DONNÉES
    client = bigquery.Client(project=ALPHA_CFG['PROJECT'])
    query = f"""
        SELECT * FROM `{ALPHA_CFG['DB_SET']}.{ALPHA_CFG['TBL']}` 
        WHERE Ticker IN ('ORA.PA','{ALPHA_CFG['IDX']}') 
        ORDER BY Date ASC
    """
    raw_df = client.query(query).to_dataframe()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.tz_localize(None)
    
    # Séparation et alignement temporel
    base_ora = raw_df[raw_df['Ticker']=='ORA.PA'].set_index('Date').copy()
    base_idx = raw_df[raw_df['Ticker']==ALPHA_CFG['IDX']].set_index('Date').copy()
    
    for df in [base_ora, base_idx]:
        for c in ['Close','High','Low','Volume']: 
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 2. CALCULS VECTORISÉS (SANS REGARD VERS LE FUTUR)
    # Force Relative (J vs J-62)
    ora_perf = base_ora['Close'].pct_change(62)
    idx_perf = base_idx['Close'].reindex(base_ora.index).pct_change(62)
    rs_line = (ora_perf - idx_perf) * 100

    # RSI Gold (Somme simple sur 14j)
    diff = base_ora['Close'].diff()
    gains = diff.clip(lower=0).rolling(14).sum()
    losses = (-diff.clip(upper=0)).rolling(14).sum()
    rsi_gold = 100 - (100 / (1 + (gains / losses)))

    # Volatilité & Moyennes
    v_ratio = base_ora['Volume'] / base_ora['Volume'].rolling(20).mean()
    mm20 = base_ora['Close'].rolling(20).mean()
    dist_mm20 = (base_ora['Close'] - mm20).abs() / mm20
    
    # Squeeze & ATR (Vectorisé avant la boucle pour la performance)
    std20 = base_ora['Close'].rolling(20).std(ddof=0)
    tr = pd.concat([
        base_ora['High'] - base_ora['Low'],
        (base_ora['High'] - base_ora['Close'].shift()).abs(),
        (base_ora['Low'] - base_ora['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr_vec = tr.rolling(ALPHA_CFG['ATR_P']).mean()
    atr_sqz = tr.rolling(20).mean()
    is_sqz = (2 * std20 < 1.5 * atr_sqz)

    # Structure (Pivots confirmés à T-3)
    roll_high = base_ora['High'].rolling(window=7, center=True).max().shift(3)
    roll_low = base_ora['Low'].rolling(window=7, center=True).min().shift(3)
    confirmed_h = (base_ora['High'].shift(3) == roll_high)
    confirmed_l = (base_ora['Low'].shift(3) == roll_low)

    # 3. CALCUL DU SCORE (RÉ-INDEXÉ EN PANDAS SERIES)
    s_val = pd.Series(0, index=base_ora.index)
    s_val += np.where((rsi_gold >= 50) & (rsi_gold <= 70), 15, 0)
    s_val += np.where(v_ratio > 1.5, 25, np.where(v_ratio > 1.1, 12.5, 0))
    s_val += np.where(dist_mm20 <= 0.01, np.where(base_ora['Close'] >= mm20, 20, -10), 0)
    s_val += np.where(is_sqz, 10, 0)
    s_val += np.where(confirmed_h & confirmed_l, 30, 0)

    # Application du verrou RS (On force le retour en Series avec index)
    score = pd.Series(np.where(rs_line <= 0, 0, s_val), index=base_ora.index)

    # 4. FILTRE MARCHÉ & PENTE
    idx_sma = base_idx['Close'].rolling(ALPHA_CFG['SMA_P']).mean().reindex(base_ora.index)
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4))
    mkt_ok = (base_idx['Close'].reindex(base_ora.index) >= idx_sma) if ALPHA_CFG['MKT_FILTER'] else pd.Series(True, index=base_ora.index)

    # 5. MOTEUR D'EXÉCUTION
    ledger = []
    active_trade = None
    
    for date, row in base_ora.iterrows():
        if active_trade:
            # Gestion de la position ouverte
            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low'] - active_trade['e_px']) / active_trade['e_px']
            
            if not active_trade['be_hit'] and h_perf >= active_trade['be_trig']:
                active_trade['be_hit'] = True
            
            current_sl = ALPHA_CFG['FEES'] if active_trade['be_hit'] else -ALPHA_CFG['STOP_L']
            
            hit_tp = h_perf >= active_trade['tp_val']
            hit_sl = l_perf <= current_sl
            
            if hit_tp or hit_sl:
                # Priorité au TP si les deux sont touchés dans la même bougie
                final_perf = active_trade['tp_val'] if (hit_tp and not (hit_sl and l_perf < current_sl)) else current_sl
                ledger.append({
                    'Achat': active_trade['date'],
                    'Vente': date,
                    'Gain': (final_perf - ALPHA_CFG['FEES']) * ALPHA_CFG['SIZE'],
                    'Status': 'TP' if hit_tp else ('BE' if active_trade['be_hit'] else 'SL')
                })
                active_trade = None
            continue

        # Check Signal d'Entrée
        if mkt_ok.loc[date] and score.loc[date] >= ALPHA_CFG['MIN_SCORE']:
            vol_pct = atr_vec.loc[date] / row['Close']
            
            active_trade = {
                'date': date,
                'e_px': row['Close'],
                'tp_val': ALPHA_CFG['TP_TREND'] if idx_slope.loc[date] >= ALPHA_CFG['SLOPE_TRESH'] else ALPHA_CFG['TP_RANGE'],
                'be_trig': ALPHA_CFG['BE_F'] if (idx_slope.loc[date] >= 0.004 and vol_pct < ALPHA_CFG['VOL_LIM']) else ALPHA_CFG['BE_S'],
                'be_hit': False
            }

    return pd.DataFrame(ledger)

# Exécution et Visualisation rapide
df_results = alpha_engine_v3()
if not df_results.empty:
    print(f"--- RÉSULTATS ALPHA ENGINE ---")
    print(f"Profit Total : {df_results['Gain'].sum():.2f} €")
    print(f"Nombre de Trades : {len(df_results)}")
    print(f"Win Rate : {(len(df_results[df_results['Gain'] > 0]) / len(df_results) * 100):.2f}%")
else:
    print("Aucun signal détecté sur la période.")
