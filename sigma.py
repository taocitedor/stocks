import pandas as pd
import numpy as np
from google.cloud import bigquery
import json

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

# === Pivots stricts (équivalent GAS) =========================================
def gas_pivots_strict(high: pd.Series, low: pd.Series, w: int = 3):
    """
    Pivot H à i si: high[i] > max(high[i-w : i-1]) ET high[i] > max(high[i+1 : i+w])
    Pivot L à i si: low[i]  < min(low[i-w : i-1])  ET low[i]  < min(low[i+1 : i+w])
    Aucun shift global; strict (pas d'égalité); False aux bords.
    """
    if not high.index.equals(low.index):
        raise ValueError("high et low doivent partager le même index.")

    # Fenêtre gauche (passé) - exclut la bougie i via shift(1)
    left_max = high.shift(1).rolling(w, min_periods=w).max()
    left_min = low.shift(1).rolling(w, min_periods=w).min()

    # Fenêtre droite (futur) - inversion pour exclure i via shift(1)
    right_max = high[::-1].shift(1).rolling(w, min_periods=w).max()[::-1]
    right_min = low[::-1].shift(1).rolling(w, min_periods=w).min()[::-1]

    is_pivot_h = (high > left_max) & (high > right_max)
    is_pivot_l = (low  < left_min) & (low  < right_min)

    return is_pivot_h.fillna(False), is_pivot_l.fillna(False)


def alpha_engine_v3():
    # 1. ACQUISITION & NETTOYAGE
    client = bigquery.Client(project=ALPHA_CFG['PROJECT'])
    query = f"""
        SELECT * FROM `{ALPHA_CFG['DB_SET']}.{ALPHA_CFG['TBL']}` 
        WHERE Ticker IN ('EN.PA','{ALPHA_CFG['IDX']}') 
        ORDER BY Date ASC
    """
    raw_df = client.query(query).to_dataframe()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.tz_localize(None)
    
    base_ora = raw_df[raw_df['Ticker']=='EN.PA'].set_index('Date').copy()
    base_idx = raw_df[raw_df['Ticker']==ALPHA_CFG['IDX']].set_index('Date').copy()
    
    for df in [base_ora, base_idx]:
        for c in ['Close','High','Low','Volume']: 
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 2. CALCULS INDICATEURS VECTORISÉS
    # Force Relative (63 séances, alignée sur ORA)
    ora_perf = base_ora['Close'].pct_change(ALPHA_CFG['LOOKBACK'])
    idx_perf = base_idx['Close'].reindex(base_ora.index).pct_change(ALPHA_CFG['LOOKBACK'])
    rs_line = (ora_perf - idx_perf) * 100

    # RSI (méthode simple somme gains/pertes sur 14)
    diff = base_ora['Close'].diff()
    gains = diff.clip(lower=0).rolling(14).sum()
    losses = (-diff.clip(upper=0)).rolling(14).sum()
    rsi_gold = 100 - (100 / (1 + (gains / losses.replace(0, np.nan))))
    rsi_gold = rsi_gold.fillna(0)

    # Volumes, MM20, distance relative
    v_ratio = (base_ora['Volume'] / base_ora['Volume'].rolling(20).mean()).fillna(0)
    mm20 = base_ora['Close'].rolling(20).mean()
    dist_mm20 = ((base_ora['Close'] - mm20).abs() / mm20).fillna(1)
    
    # Squeeze (proche GAS: 2*std20 < 1.5*ATR20_HL)
    hl_range = base_ora['High'] - base_ora['Low']
    atr_sqz = hl_range.rolling(20).mean().fillna(0)
    std20 = base_ora['Close'].rolling(20).std(ddof=0)
    is_sqz = (2 * std20 < 1.5 * atr_sqz).fillna(False)

    # --- ALIGNEMENT STRUCTURE GAS (Séquence HH + HL) ---
    is_pivot_h, is_pivot_l = gas_pivots_strict(base_ora['High'], base_ora['Low'], w=3)

    # Valeurs aux pivots (NaN ailleurs)
    p_h_vals = base_ora['High'].where(is_pivot_h)
    p_l_vals = base_ora['Low'] .where(is_pivot_l)

    # Derniers pivots confirmés (h1/l1) et précédents (h2/l2)
    last_h1 = p_h_vals.ffill()
    last_h2 = p_h_vals.dropna().shift(1).reindex(base_ora.index).ffill()
    last_l1 = p_l_vals.ffill()
    last_l2 = p_l_vals.dropna().shift(1).reindex(base_ora.index).ffill()

    # Structure HH + HL (avec masque de validité: au moins 2 H et 2 L)
    is_hh = last_h1 > last_h2
    is_hl = last_l1 > last_l2
    valid_struct = last_h2.notna() & last_l2.notna()
    struct_ok = ((is_hh & is_hl) & valid_struct).fillna(False)

    # 3. CALCUL DU SCORE (mêmes pondérations que GAS)
    s_val = pd.Series(0.0, index=base_ora.index)
    s_val += np.where((rsi_gold >= 50) & (rsi_gold <= 70), 15, 0)
    s_val += np.where(v_ratio > 1.5, 25, np.where(v_ratio > 1.1, 12.5, 0))
    s_val += np.where(dist_mm20 <= 0.01, np.where(base_ora['Close'] >= mm20, 20, -10), 0)
    s_val += np.where(is_sqz, 10, 0)
    s_val += np.where(struct_ok, 30, 0)

    score = pd.Series(np.where(rs_line.fillna(-1) <= 0, 0, s_val), index=base_ora.index)

    # 4. FILTRE MARCHÉ & ATR D'ENTRÉE
    idx_sma = base_idx['Close'].rolling(ALPHA_CFG['SMA_P']).mean().reindex(base_ora.index)
    # pente SMA sur 4 séances (comme GAS: smaCurr vs smaPrev décalée de 4)
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4)).fillna(0)
    # filtre marché strict (close > SMA)
    mkt_ok = (base_idx['Close'].reindex(base_ora.index) > idx_sma).fillna(False) if ALPHA_CFG['MKT_FILTER'] else pd.Series(True, index=base_ora.index)

    # ATR classique (True Range), moyen sur ATR_P et décalé d'1 jour (pas d'info future)
    tr = pd.concat([
        base_ora['High'] - base_ora['Low'],
        (base_ora['High'] - base_ora['Close'].shift()).abs(),
        (base_ora['Low']  - base_ora['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr_vec = tr.rolling(ALPHA_CFG['ATR_P']).mean().shift(1).fillna(0)

    # 5. MOTEUR DE TRADING
    ledger = []
    active_trade = None

    # ====== BLOC DE DEBUG (facultatif) =======================================
    cible = pd.to_datetime('2025-07-30')
    df_debug = pd.DataFrame({
        'Close': base_ora['Close'].round(2),
        'Mkt_OK (Filtre SMA)': mkt_ok,
        'RS_Line (>0)': rs_line.round(2),
        'RSI (50-70)': rsi_gold.round(2),
        'Vol_Ratio (>1.1/1.5)': v_ratio.round(2),
        'Dist_MM20 (<=1%)': dist_mm20.round(4),
        'Squeeze (True)': is_sqz,
        'Structure (HH+HL)': struct_ok,
        'SCORE FINAL': pd.Series(score).round(2)
    })
    try:
        idx_loc = abs(df_debug.index - cible).argmin()
        print(f"\n--- 🕵️ ANALYSE DE LA ZONE DU {cible.strftime('%Y-%m-%d')} ---")
        print(df_debug.iloc[max(0, idx_loc-3): idx_loc+4].to_string())
        print("--------------------------------------------------\n")
    except Exception as e:
        print(f"Erreur lors de l'affichage : {e}")
    # ========================================================================

    for date in base_ora.index[ALPHA_CFG['SMA_P']:]:
        row = base_ora.loc[date]
        
        if active_trade:
            # Perf intra-barre relative au prix d'entrée
            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low']  - active_trade['e_px']) / active_trade['e_px']

            # 🔒 BE "bar+1": on constate le déclenchement, mais on ne l'applique
            # qu'après avoir évalué TP/SL avec l'état antérieur.
            beTriggeredThisBar = (not active_trade['be_hit']) and (h_perf >= active_trade['be_trig'])

            effectiveSL = ALPHA_CFG['FEES'] if active_trade['be_hit'] else -ALPHA_CFG['STOP_L']
            hit_tp = h_perf >= active_trade['tp_val']
            hit_sl = l_perf <= effectiveSL

            if hit_tp or hit_sl:
                # Règle pessimiste: si TP & SL touchés, on considère le SL (ordre intra-barre inconnu)
                raw_exit = effectiveSL if hit_sl else active_trade['tp_val']
                gain_cash = (raw_exit - ALPHA_CFG['FEES']) * ALPHA_CFG['SIZE']
                ledger.append({
                    'Achat': active_trade['date'].strftime('%Y-%m-%d'),
                    'Vente': date.strftime('%Y-%m-%d'),
                    'Gain': float(gain_cash),
                    'Type': 'TP' if (hit_tp and not hit_sl) else ('BE' if active_trade['be_hit'] else 'SL')
                })
                active_trade = None
            else:
                # On active le BE seulement en FIN de bougie
                if beTriggeredThisBar:
                    active_trade['be_hit'] = True
            continue

        # ---- Check d'entrée (mêmes conditions que GAS) ----
        if mkt_ok.loc[date] and score.loc[date] >= ALPHA_CFG['MIN_SCORE']:
            vol_pct = float(atr_vec.loc[date] / row['Close'])
            active_trade = {
                'date': date,
                'e_px': float(row['Close']),  # Si tu as 'Open', je bascule ici vers next-bar open
                'tp_val': ALPHA_CFG['TP_TREND'] if idx_slope.loc[date] >= ALPHA_CFG['SLOPE_TRESH'] else ALPHA_CFG['TP_RANGE'],
                'be_trig': ALPHA_CFG['BE_F'] if (idx_slope.loc[date] >= 0.004 and vol_pct < ALPHA_CFG['VOL_LIM']) else ALPHA_CFG['BE_S'],
                'be_hit': False
            }

    # 6. FORMATAGE JSON
    df_ledger = pd.DataFrame(ledger)
    res = {
        "metadata": {"system": "Alpha Engine v3.2", "ticker": "EN.PA"},
        "performance": {
            "gain_total": float(df_ledger['Gain'].sum()) if not df_ledger.empty else 0.0,
            "nb_trades": int(len(df_ledger)),
            "win_rate": float(len(df_ledger[df_ledger['Gain'] > 0]) / len(df_ledger)) if not df_ledger.empty else 0.0
        },
        "trades": df_ledger.to_dict(orient='records') if not df_ledger.empty else []
    }
    return res


if __name__ == "__main__":
    results = alpha_engine_v3()
    print(json.dumps(results, indent=2))
