import pandas as pd
import numpy as np
from google.cloud import bigquery

# --- CONFIGURATION ALPHA ---
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

def sigma_engine():
    client = bigquery.Client(project=ALPHA_CFG['PROJECT'])
    
    # 1. FETCH & ALIGNMENT
    query = f"SELECT * FROM `{ALPHA_CFG['DB_SET']}.{ALPHA_CFG['TBL']}` WHERE Ticker IN ('ORA.PA','{ALPHA_CFG['IDX']}') ORDER BY Date ASC"
    raw_df = client.query(query).to_dataframe()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.tz_localize(None)
    
    # Split et Indexation par date pour alignement parfait
    base_ora = raw_df[raw_df['Ticker']=='ORA.PA'].set_index('Date').copy()
    base_idx = raw_df[raw_df['Ticker']==ALPHA_CFG['IDX']].set_index('Date').copy()
    
    for df in [base_ora, base_idx]:
        for c in ['Close','High','Low','Volume']: df[c] = df[c].astype(float)

    # 2. INDICATEURS VECTORISÉS (SANS LOOK-AHEAD)
    # Force Relative (J vs J-62)
    ora_perf = base_ora['Close'].pct_change(62)
    idx_perf = base_idx['Close'].reindex(base_ora.index).pct_change(62)
    rs_line = (ora_perf - idx_perf) * 100

    # RSI Gold (Somme simple sur 14j, décalé de 1 pour ne pas utiliser le Close actuel dans le signal)
    diff = base_ora['Close'].diff()
    gains = diff.clip(lower=0).rolling(14).sum()
    losses = (-diff.clip(upper=0)).rolling(14).sum()
    rsi_gold = 100 - (100 / (1 + (gains / losses)))

    # Volatilité & Moyennes
    v_ratio = base_ora['Volume'] / base_ora['Volume'].rolling(20).mean()
    mm20 = base_ora['Close'].rolling(20).mean()
    dist_mm20 = (base_ora['Close'] - mm20).abs() / mm20
    
    # Squeeze (Std Dev Pop)
    std20 = base_ora['Close'].rolling(20).std(ddof=0)
    atr20_sqz = (base_ora['High'] - base_ora['Low']).rolling(20).mean()
    is_sqz = (2 * std20 < 1.5 * atr20_sqz)

    # Structure (Pivots confirmés : on regarde le passé uniquement)
    # Un pivot à T-3 est confirmé si T-3 est le point extrême sur [T-6, T]
    roll_high = base_ora['High'].rolling(window=7, center=True).max()
    roll_low = base_ora['Low'].rolling(window=7, center=True).min()
    # On décale de 3 pour que la confirmation (center) ne lise pas le futur
    confirmed_h = (base_ora['High'].shift(3) == roll_high.shift(3))
    confirmed_l = (base_ora['Low'].shift(3) == roll_low.shift(3))

    # 3. CALCUL DU SCORE SIGMA (MATRICIEL)
    score = pd.Series(0, index=base_ora.index)
    score += np.where((rsi_gold >= 50) & (rsi_gold <= 70), 15, 0)
    score += np.where(v_ratio > 1.5, 25, np.where(v_ratio > 1.1, 12.5, 0))
    score += np.where(dist_mm20 <= 0.01, np.where(base_ora['Close'] >= mm20, 20, -10), 0)
    score += np.where(is_sqz, 10, 0)
    
    # Structure Points (Approximation vectorielle HH/HL sur pivots confirmés)
    score += np.where(confirmed_h & confirmed_l, 30, 0)

    # Verrou de Force Relative (Crucial)
    score = np.where(rs_line <= 0, 0, score)

    # 4. FILTRE MARCHÉ & SLOPE
    idx_sma = base_idx['Close'].rolling(ALPHA_CFG['SMA_P']).mean().reindex(base_ora.index)
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4))
    mkt_ok = (base_idx['Close'].reindex(base_ora.index) >= idx_sma) if ALPHA_CFG['MKT_FILTER'] else True

    # 5. ENGINE DE TRADING (LOOP OPTIMISÉ)
    ledger = []
    active_trade = None
    
    # On itère car la gestion du TP/SL/BE est interdépendante du prix d'entrée
    for date, row in base_ora.iterrows():
        if active_trade:
            # Check Sortie
            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low'] - active_trade['e_px']) / active_trade['e_px']
            
            if not active_trade['be_hit'] and h_perf >= active_trade['be_trig']:
                active_trade['be_hit'] = True
            
            current_sl = ALPHA_CFG['FEES'] if active_trade['be_hit'] else -ALPHA_CFG['STOP_L']
            
            hit_tp = h_perf >= active_trade['tp_val']
            hit_sl = l_perf <= current_sl
            
            if hit_tp or hit_sl:
                final_perf = active_trade['tp_val'] if (hit_tp and not (hit_sl and l_perf < current_sl)) else current_sl
                ledger.append({
                    'Entry': active_trade['date'],
                    'Exit': date,
                    'Net_Gain': (final_perf - ALPHA_CFG['FEES']) * ALPHA_CFG['SIZE']
                })
                active_trade = None
            continue

        # Check Entrée
        if mkt_ok.loc[date] and score.loc[date] >= ALPHA_CFG['MIN_SCORE']:
            # Calcul ATR Volatilité pour BE
            tr = pd.concat([
                base_ora['High'] - base_ora['Low'],
                (base_ora['High'] - base_ora['Close'].shift()).abs(),
                (base_ora['Low'] - base_ora['Close'].shift()).abs()
            ], axis=1).max(axis=1)
            atr_val = tr.loc[:date].tail(ALPHA_CFG['ATR_P']).mean()
            vol_pct = atr_val / row['Close']

            active_trade = {
                'date': date,
                'e_px': row['Close'],
                'tp_val': ALPHA_CFG['TP_TREND'] if idx_slope.loc[date] >= ALPHA_CFG['SLOPE_TRESH'] else ALPHA_CFG['TP_RANGE'],
                'be_trig': ALPHA_CFG['BE_F'] if (idx_slope.loc[date] >= 0.004 and vol_pct < ALPHA_CFG['VOL_LIM']) else ALPHA_CFG['BE_S'],
                'be_hit': False
            }

    return pd.DataFrame(ledger)

# Exécution
res = sigma_engine()
print(f"Gain Total: {res['Net_Gain'].sum():.2f} €")
