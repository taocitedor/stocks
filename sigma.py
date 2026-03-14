# ==== START sigma.py
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery

ALPHA_CFG = {
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours',
    'STOCK': 'EN.PA',
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
    'SIZE': 4000,
    'PIVOT_W': 3,
    'STRUCT_LAST_PIVOTS': 15,
    'DEBUG_DATE': '2025-07-29',
}


def gas_rs_series(stock_close: pd.Series, idx_close: pd.Series) -> pd.Series:
    """
    RS alignée GAS :
      - besoin d'au moins 63 barres côté titre
      - stock_perf = close[t] vs close[t-62]
      - pour l'indice : dernier close <= date du titre, puis 62 barres plus tôt
      - RS = (stock_perf - idx_perf) * 100
    """
    stock_close = pd.to_numeric(stock_close, errors='coerce').sort_index()
    idx_close = pd.to_numeric(idx_close, errors='coerce').sort_index()

    s_dates = stock_close.index.to_numpy()
    s_vals = stock_close.to_numpy(dtype=float)
    i_dates = idx_close.index.to_numpy()
    i_vals = idx_close.to_numpy(dtype=float)

    out = np.zeros(len(stock_close), dtype=float)

    for i in range(len(stock_close)):
        if i < 62:
            out[i] = 0.0
            continue

        current_dt = s_dates[i]
        pos = np.searchsorted(i_dates, current_dt, side='right') - 1
        if pos < 62:
            out[i] = 0.0
            continue

        s_curr = s_vals[i]
        s_prev = s_vals[i - 62]
        i_curr = i_vals[pos]
        i_prev = i_vals[pos - 62]

        if (
            np.isnan(s_curr) or np.isnan(s_prev) or s_prev == 0 or
            np.isnan(i_curr) or np.isnan(i_prev) or i_prev == 0
        ):
            out[i] = 0.0
            continue

        stock_perf = (s_curr - s_prev) / s_prev
        idx_perf = (i_curr - i_prev) / i_prev
        out[i] = (stock_perf - idx_perf) * 100.0

    return pd.Series(out, index=stock_close.index, name='RS_Line')


def gas_rsi_series(close: pd.Series, p: int = 14) -> pd.Series:
    """
    RSI façon GAS (somme gains/pertes sur p barres ; pertes==0 -> 100)
    """
    close = pd.to_numeric(close, errors='coerce')
    diff = close.diff()
    gains = diff.clip(lower=0).rolling(p, min_periods=p).sum()
    losses = (-diff.clip(upper=0)).rolling(p, min_periods=p).sum()

    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    zero_losses = losses == 0
    normal_mask = (~zero_losses) & losses.notna()

    rsi[zero_losses] = 100.0
    rsi[normal_mask] = 100.0 - (100.0 / (1.0 + (gains[normal_mask] / losses[normal_mask])))

    return rsi.fillna(0.0)


def gas_pivots_events(df: pd.DataFrame, w: int = 3):
    """
    Pivots stricts comme GAS :
      - H si aucun voisin (i-w..i+w, hors i) n'a high >= high[i]
      - L si aucun voisin (i-w..i+w, hors i) n'a low  <= low[i]
    """
    highs = pd.to_numeric(df['High'], errors='coerce').to_numpy(dtype=float)
    lows  = pd.to_numeric(df['Low'],  errors='coerce').to_numpy(dtype=float)

    pivots = []
    n = len(df)

    for i in range(w, n - w):
        is_h = True
        is_l = True
        for j in range(i - w, i + w + 1):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                is_h = False
            if lows[j]  <= lows[i]:
                is_l = False
            if not is_h and not is_l:
                break

        if is_h:
            pivots.append({'pivot_i': i, 'type': 'H', 'value': float(highs[i])})
        if is_l:
            pivots.append({'pivot_i': i, 'type': 'L', 'value': float(lows[i])})

    return pivots


def gas_structure_series(df: pd.DataFrame, w: int = 3, last_pivots: int = 15):
    """
    Structure comme GAS :
      - un pivot détecté à k devient visible à k+w
      - à chaque date i, on prend les pivots visibles, on slice(-15)
      - HH/LH sur les deux derniers H ; HL/LL sur les deux derniers L
    """
    df = df.sort_index().copy()
    n = len(df)
    all_pivots = gas_pivots_events(df, w=w)

    visible_on = [[] for _ in range(n)]
    for p in all_pivots:
        vis_i = p['pivot_i'] + w
        if vis_i < n:
            visible_on[vis_i].append(p)

    active_pivots = []
    struct_label = np.array(['ND'] * n, dtype=object)
    struct_ok = np.zeros(n, dtype=bool)

    for i in range(n):
        if visible_on[i]:
            active_pivots.extend(visible_on[i])

        p15 = active_pivots[-last_pivots:]
        highs = [x for x in p15 if x['type'] == 'H']
        lows  = [x for x in p15 if x['type'] == 'L']

        if len(highs) < 2 or len(lows) < 2:
            struct_label[i] = 'ND'
            struct_ok[i] = False
            continue

        struct = (
            ('HH' if highs[-1]['value'] > highs[-2]['value'] else 'LH')
            + '+'
            + ('HL' if lows[-1]['value'] > lows[-2]['value'] else 'LL')
        )
        struct_label[i] = struct
        struct_ok[i] = ('HH' in struct and 'HL' in struct)

    return (
        pd.Series(struct_label, index=df.index, name='Structure'),
        pd.Series(struct_ok, index=df.index, name='Structure_OK'),
    )


def gas_squeeze_series(df: pd.DataFrame) -> pd.Series:
    """
    2*std20(population) < 1.5*avg20(high-low)
    """
    close = pd.to_numeric(df['Close'], errors='coerce')
    hl    = pd.to_numeric(df['High'],  errors='coerce') - pd.to_numeric(df['Low'], errors='coerce')
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    atr20 = hl.rolling(20, min_periods=20).mean()
    return ((2.0 * std20) < (1.5 * atr20)).fillna(False)


def atr_true_range_series(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df['High'], errors='coerce')
    low  = pd.to_numeric(df['Low'],  errors='coerce')
    prev_close = pd.to_numeric(df['Close'], errors='coerce').shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def alpha_engine_v3():
    # 1) DATA
    client = bigquery.Client(project=ALPHA_CFG['PROJECT'])
    query = f"""
        SELECT *
        FROM `{ALPHA_CFG['DB_SET']}.{ALPHA_CFG['TBL']}`
        WHERE Ticker IN ('{ALPHA_CFG['STOCK']}', '{ALPHA_CFG['IDX']}')
        ORDER BY Date ASC
    """
    raw_df = client.query(query).to_dataframe()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.tz_localize(None)

    base_stock = raw_df[raw_df['Ticker'] == ALPHA_CFG['STOCK']].set_index('Date').copy()
    base_idx   = raw_df[raw_df['Ticker'] == ALPHA_CFG['IDX']].set_index('Date').copy()

    for df in [base_stock, base_idx]:
        for c in ['Close', 'High', 'Low', 'Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    base_stock = base_stock.sort_index()
    base_idx   = base_idx.sort_index()

    # 2) INDICATEURS
    rs_line  = gas_rs_series(base_stock['Close'], base_idx['Close'])
    rsi_gold = gas_rsi_series(base_stock['Close'], p=14)
    v_ratio  = (base_stock['Volume'] / base_stock['Volume'].rolling(20, min_periods=20).mean()).fillna(0.0)
    mm20     = base_stock['Close'].rolling(20, min_periods=20).mean()
    dist_mm20 = ((base_stock['Close'] - mm20).abs() / mm20).fillna(1.0)
    is_sqz   = gas_squeeze_series(base_stock)
    struct_label, struct_ok = gas_structure_series(
        base_stock, w=ALPHA_CFG['PIVOT_W'], last_pivots=ALPHA_CFG['STRUCT_LAST_PIVOTS']
    )

    # 3) SCORE
    s_val = pd.Series(0.0, index=base_stock.index)
    s_val += np.where((rsi_gold >= 50) & (rsi_gold <= 70), 15, 0)
    s_val += np.where(v_ratio > 1.5, 25, np.where(v_ratio > 1.1, 12.5, 0))
    s_val += np.where(dist_mm20 <= 0.01, np.where(base_stock['Close'] >= mm20, 20, -10), 0)
    s_val += np.where(struct_ok, 30, 0)
    s_val += np.where(is_sqz, 10, 0)
    score = pd.Series(np.where(rs_line <= 0, 0, s_val), index=base_stock.index)

    # 4) FILTRE MARCHÉ & ATR
    idx_close_on_stock_dates = base_idx['Close'].reindex(base_stock.index)
    idx_sma  = base_idx['Close'].rolling(ALPHA_CFG['SMA_P'], min_periods=ALPHA_CFG['SMA_P']).mean()
    idx_sma_on_stock_dates = idx_sma.reindex(base_stock.index)
    idx_slope = ((idx_sma_on_stock_dates - idx_sma_on_stock_dates.shift(4)) / idx_sma_on_stock_dates.shift(4)).fillna(0.0)
    mkt_ok = (idx_close_on_stock_dates > idx_sma_on_stock_dates).fillna(False) if ALPHA_CFG['MKT_FILTER'] else \
             pd.Series(True, index=base_stock.index)

    tr = atr_true_range_series(base_stock)
    atr_vec = tr.rolling(ALPHA_CFG['ATR_P'], min_periods=ALPHA_CFG['ATR_P']).mean().shift(1).fillna(0.0)

    # 5) DEBUG (sans try/except)
    cible = pd.to_datetime(ALPHA_CFG['DEBUG_DATE'])
    df_debug = pd.DataFrame({
        'Close': base_stock['Close'].round(2),
        'Mkt_OK (Filtre SMA)': mkt_ok,
        'RS_Line (>0)': rs_line.round(2),
        'RSI (50-70)': rsi_gold.round(2),
        'Vol_Ratio (>1.1/1.5)': v_ratio.round(2),
        'Dist_MM20 (<=1%)': dist_mm20.round(4),
        'Squeeze (True)': is_sqz,
        'Structure': struct_label,
        'Structure (HH+HL)': struct_ok,
        'SCORE FINAL': score.round(2),
    })
    idx_loc = int(np.argmin(np.abs(df_debug.index - cible)))
    print(f"\n--- ANALYSE DE LA ZONE DU {cible.strftime('%Y-%m-%d')} ---")
    print(df_debug.iloc[max(0, idx_loc - 3): idx_loc + 4].to_string())
    print("--------------------------------------------------\n")

    # 6) MOTEUR TRADING
    ledger = []
    active_trade = None

    for date in base_stock.index[ALPHA_CFG['SMA_P']:]:
        row = base_stock.loc[date]

        if active_trade is not None:
            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low']  - active_trade['e_px']) / active_trade['e_px']
            be_triggered_this_bar = (not active_trade['be_hit']) and (h_perf >= active_trade['be_trig'])

            effective_sl = ALPHA_CFG['FEES'] if active_trade['be_hit'] else -ALPHA_CFG['STOP_L']
            hit_tp = h_perf >= active_trade['tp_val']
            hit_sl = l_perf <= effective_sl

            if hit_tp or hit_sl:
                raw_exit = effective_sl if hit_sl else active_trade['tp_val']
                gain_cash = (raw_exit - ALPHA_CFG['FEES']) * ALPHA_CFG['SIZE']
                trade_type = 'TP' if (hit_tp and not hit_sl) else ('BE' if active_trade['be_hit'] else 'SL')
                ledger.append({
                    'Achat': active_trade['date'].strftime('%Y-%m-%d'),
                    'Vente': date.strftime('%Y-%m-%d'),
                    'Gain': float(gain_cash),
                    'Type': trade_type,
                })
                active_trade = None
            else:
                if be_triggered_this_bar:
                    active_trade['be_hit'] = True
            continue

        if bool(mkt_ok.loc[date]) and float(score.loc[date]) >= ALPHA_CFG['MIN_SCORE']:
            vol_pct = float(atr_vec.loc[date] / row['Close']) if pd.notna(row['Close']) and row['Close'] != 0 else 0.0
            active_trade = {
                'date': date,
                'e_px': float(row['Close']),  # pour exécution next-bar open, remplace ici si tu as 'Open'
                'tp_val': ALPHA_CFG['TP_TREND'] if idx_slope.loc[date] >= ALPHA_CFG['SLOPE_TRESH'] else ALPHA_CFG['TP_RANGE'],
                'be_trig': ALPHA_CFG['BE_F'] if (idx_slope.loc[date] >= 0.004 and vol_pct < ALPHA_CFG['VOL_LIM']) else ALPHA_CFG['BE_S'],
                'be_hit': False,
            }

    # 7) SORTIE
    df_ledger = pd.DataFrame(ledger)
    return {
        'metadata': {
            'system': 'Alpha Engine v3.3 GAS-parity',
            'ticker': ALPHA_CFG['STOCK'],
        },
        'performance': {
            'gain_total': float(df_ledger['Gain'].sum()) if not df_ledger.empty else 0.0,
            'nb_trades': int(len(df_ledger)),
            'win_rate': float((df_ledger['Gain'] > 0).mean()) if not df_ledger.empty else 0.0,
        },
        'trades': df_ledger.to_dict(orient='records') if not df_ledger.empty else [],
    }


if __name__ == '__main__':
    results = alpha_engine_v3()
    print(json.dumps(results, indent=2, ensure_ascii=False))
# ==== END sigma.py
