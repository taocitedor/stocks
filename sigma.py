import pandas as pd
import numpy as np
from google.cloud import bigquery
import json

# --- CONFIGURATION SYSTÈME ALPHA (CONSTANTES GOLD STANDARD) ---
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
    # 1. ACQUISITION & NETTOYAGE
    client = bigquery.Client(project=ALPHA_CFG['PROJECT'])
    query = f"""
        SELECT * FROM `{ALPHA_CFG['DB_SET']}.{ALPHA_CFG['TBL']}` 
        WHERE Ticker IN ('ORA.PA','{ALPHA_CFG['IDX']}') 
        ORDER BY Date ASC
    """
    raw_df = client.query(query).to_dataframe()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.tz_localize(None)
    
    # Séparation ORA / CAC40
    base_ora = raw_df[raw_df['Ticker']=='ORA.PA'].set_index('Date').copy()
    base_idx = raw_df[raw_df['Ticker']==ALPHA_CFG['IDX']].set_index('Date').copy()
    
    for df in [base_ora, base_idx]:
        for c in ['Close','High','Low','Volume']: 
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 2. CALCULS INDICATEURS VECTORISÉS (SANS REGARD VERS LE FUTUR)
    # Force Relative (J vs J-62)
    ora_perf = base_ora['Close'].pct_change(62)
    idx_perf = base_idx['Close'].reindex(base_ora.index).pct_change(62)
    rs_line = (ora_perf - idx_perf) * 100

    # RSI Gold (Somme simple 14j) - Sécurité division par zéro
    diff = base_ora['Close'].diff()
    gains = diff.clip(lower=0).rolling(14).sum()
    losses = (-diff.clip(upper=0)).rolling(14).sum()
    rsi_gold = 100 - (100 / (1 + (gains / losses.replace(0, np.nan))))
    rsi_gold = rsi_gold.fillna(0)

    # Volatilité & Moyennes
    v_ratio = (base_ora['Volume'] / base_ora['Volume'].rolling(20).mean()).fillna(0)
    mm20 = base_ora['Close'].rolling(20).mean()
    dist_mm20 = ((base_ora['Close'] - mm20).abs() / mm20).fillna(1)
    
    # ATR & Squeeze (Vectorisé pour la performance)
    std20 = base_ora['Close'].rolling(20).std(ddof=0)
    tr = pd.concat([
        base_ora['High'] - base_ora['Low'],
        (base_ora['High'] - base_ora['Close'].shift()).abs(),
        (base_ora['Low'] - base_ora['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr_vec = tr.rolling(ALPHA_CFG['ATR_P']).mean().fillna(0)
    atr_sqz = tr.rolling(20).mean().fillna(0)
    is_sqz = (2 * std20 < 1.5 * atr_sqz).fillna(False)

    # Structure (Pivots confirmés à T-3 pour éviter le look-ahead)
    roll_h = base_ora['High'].rolling(window=7, center=True).max().shift(3)
    roll_l = base_ora['Low'].rolling(window=7, center=True).min().shift(3)
    confirmed_h = (base_ora['High'].shift(3) == roll_h).fillna(False)
    confirmed_l = (base_ora['Low'].shift(3) == roll_l).fillna(False)

    # 3. CALCUL DU SCORE (MOTEUR MATRICIEL)
    # On reste en Series Pandas pour garder l'index des dates
    s_val = pd.Series(0, index=base_ora.index)
    s_val += np.where((rsi_gold >= 50) & (rsi_gold <= 70), 15, 0)
    s_val += np.where(v_ratio > 1.5, 25, np.where(v_ratio > 1.1, 12.5, 0))
    s_val += np.where(dist_mm20 <= 0.01, np.where(base_ora['Close'] >= mm20, 20, -10), 0)
    s_val += np.where(is_sqz, 10, 0)
    s_val += np.where(confirmed_h & confirmed_l, 30, 0)

    # Verrou RS & Reconstruction de la Series Score
    score = pd.Series(np.where(rs_line.fillna(-1) <= 0, 0, s_val), index=base_ora.index)

    # 4. FILTRE MARCHÉ & SLOPE
    idx_sma = base_idx['Close'].rolling(ALPHA_CFG['SMA_P']).mean().reindex(base_ora.index)
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4)).fillna(0)
    mkt_ok = (base_idx['Close'].reindex(base_ora.index) >= idx_sma).fillna(False) if ALPHA_CFG['MKT_FILTER'] else pd.Series(True, index=base_ora.index)

    # 5. MOTEUR DE TRADING
    ledger = []
    active_trade = None
    
    # Démarrage après la période de chauffe SMA 100
    for date in base_ora.index[ALPHA_CFG['SMA_P']:]:
        row = base_ora.loc[date]
        
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
                    'Achat': active_trade['date'].strftime('%Y-%m-%d'),
                    'Vente': date.strftime('%Y-%m-%d'),
                    'Gain': float((final_perf - ALPHA_CFG['FEES']) * ALPHA_CFG['SIZE']),
                    'Type': 'TP' if hit_tp else ('BE' if active_trade['be_hit'] else 'SL')
                })
                active_trade = None
            continue

        # Check Entrée
        if mkt_ok.loc[date] and score.loc[date] >= ALPHA_CFG['MIN_SCORE']:
            vol_pct = float(atr_vec.loc[date] / row['Close'])
            active_trade = {
                'date': date,
                'e_px': float(row['Close']),
                'tp_val': ALPHA_CFG['TP_TREND'] if idx_slope.loc[date] >= ALPHA_CFG['SLOPE_TRESH'] else ALPHA_CFG['TP_RANGE'],
                'be_trig': ALPHA_CFG['BE_F'] if (idx_slope.loc[date] >= 0.004 and vol_pct < ALPHA_CFG['VOL_LIM']) else ALPHA_CFG['BE_S'],
                'be_hit': False
            }

    # 6. FORMATAGE JSON FINAL
    df_ledger = pd.DataFrame(ledger)
    res = {
        "metadata": {"system": "Alpha Engine v3.2", "ticker": "ORA.PA"},
        "performance": {
            "gain_total": float(df_ledger['Gain'].sum()) if not df_ledger.empty else 0.0,
            "nb_trades": int(len(df_ledger)),
            "win_rate": float(len(df_ledger[df_ledger['Gain'] > 0]) / len(df_ledger)) if not df_ledger.empty else 0.0
        },
        "trades": df_ledger.to_dict(orient='records') if not df_ledger.empty else []
    }
    return res

if __name__ == "__main__":
    results = run_alpha_backtest()
    print(json.dumps(results, indent=2))
