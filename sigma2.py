# ==== START sigma2.py
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery

ALPHA4_CFG = {
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours',
    'IDX': '^FCHI',
    'MKT_FILTER': True,
    'SMA_P': 100,
    'MIN_SCORE': 86,
    'LOOKBACK': 63,

    'USE_RS_SMA_FILTER': True,   # Active le filtre de vélocité
    'RS_SMA_P': 20,              # Période de lissage de la RS

    # --- Pondération (Total 100) ---
    'W_STRUCT': 30,
    'W_VOL': 25,
    'W_DIST_M20': 20,
    'W_RSI': 15,
    'W_SQZ': 10,

    # --- Paramètres de Pénalité / Disqualification ---
    'PENALTY_MM20': -10,         # Retrait de points si Close < MM20
    'FORCE_RS_POSITIVE': True,   # Si True, RS <= 0 écrase le score à 0

    # --- Filtre tendance titre (désactivable) ---
    'USE_PRICE_SMA_FILTER': True,  # True = on exige Close >= SMA long terme
    'PRICE_SMA_P': 200,            # période de la SMA du titre

    # --- Paramètres TP (Dynamiques) ---
    'TP_TREND': 0.13,        # TP standard en tendance
    'TP_RANGE': 0.10,        # TP si marché mou/baissier
    'TP_BOOST': 0.02,        # Bonus TP si fort momentum
    'SLOPE_TRESH': -0.003,   # Seuil pente SMA marché
    'SLOPE_STRONG': 0.005,   # Seuil pour déclencher le "Boost" de performance

    # --- Paramètres BE (Adaptatifs) ---
    'BE_F': 0.06,            # BE fast (marché propre)
    'BE_S': 0.0495,          # BE slow (marché volatil)
    'BE_DELAY': 3,           # Nombre de barres minimum avant d'autoriser le BE
    'VOL_LIM': 0.025,        # Limite volatilité pour passer de BE_F à BE_S

    # --- Gestion Risque & Frais ---
    'ATR_P': 50,
    'STOP_L': 0.10,          # Stop Loss initial
    'FEES': 0.0056,          # Frais (appliqués à l'entrée et à la sortie)
    'SIZE': 4000,

    # --- Structure & Pivots ---
    'PIVOT_W': 3,
    'STRUCT_LAST_PIVOTS': 15,
    'UNIVERSE': None,        # Optionnel: restreindre l’univers, ex. ['EN.PA','ORA.PA'] ; None => tous
}

# ===========================
# Indicateurs (parité GAS)
# ===========================
def v4_rs_line(stock_close: pd.Series, idx_close: pd.Series) -> pd.Series:
    """
    RS alignée GAS :
      - besoin d'au moins 63 barres côté titre
      - stock_perf = close[t] vs close[t-62]
      - indice : dernier close <= date du titre, puis 62 barres plus tôt
      - RS = (stock_perf - idx_perf) * 100
    """
    stock_close = pd.to_numeric(stock_close, errors='coerce').sort_index()
    idx_close   = pd.to_numeric(idx_close, errors='coerce').sort_index()

    s_dates = stock_close.index.to_numpy()
    s_vals  = stock_close.to_numpy(dtype=float)
    i_dates = idx_close.index.to_numpy()
    i_vals  = idx_close.to_numpy(dtype=float)

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
        idx_perf   = (i_curr - i_prev) / i_prev
        out[i] = (stock_perf - idx_perf) * 100.0

    return pd.Series(out, index=stock_close.index, name='RS_Line')


def v4_rsi(close: pd.Series, p: int = 14) -> pd.Series:
    """
    RSI façon GAS (somme gains/pertes sur p barres ; pertes==0 -> 100)
    """
    close = pd.to_numeric(close, errors='coerce')
    diff = close.diff()
    gains = diff.clip(lower=0).rolling(p, min_periods=p).sum()
    losses = (-diff.clip(upper=0)).rolling(p, min_periods=p).sum()

    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    zero_losses = (losses == 0)
    normal_mask = (~zero_losses) & losses.notna()

    rsi[zero_losses] = 100.0
    rsi[normal_mask] = 100.0 - (100.0 / (1.0 + (gains[normal_mask] / losses[normal_mask])))
    return rsi.fillna(0.0)


def v4_pivot_events(df: pd.DataFrame, w: int = 3):
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
            if lows[j] <= lows[i]:
                is_l = False
            if not is_h and not is_l:
                break

        if is_h:
            pivots.append({'pivot_i': i, 'type': 'H', 'value': float(highs[i])})
        if is_l:
            pivots.append({'pivot_i': i, 'type': 'L', 'value': float(lows[i])})

    return pivots


def v4_structure_labels(df: pd.DataFrame, w: int = 3, last_pivots: int = 15):
    """
    Structure GAS :
      - un pivot détecté à k devient visible à k+w
      - à chaque date i, on prend les pivots visibles, on slice(-15)
      - HH/LH sur les deux derniers H ; HL/LL sur les deux derniers L
    """
    df = df.sort_index().copy()
    n = len(df)
    piv = v4_pivot_events(df, w=w)

    visible_on = [[] for _ in range(n)]
    for p in piv:
        vis_i = p['pivot_i'] + w
        if vis_i < n:
            visible_on[vis_i].append(p)

    active = []
    struct_label = np.array(['ND'] * n, dtype=object)
    struct_ok = np.zeros(n, dtype=bool)

    for i in range(n):
        if visible_on[i]:
            active.extend(visible_on[i])

        p15 = active[-last_pivots:]
        h = [x for x in p15 if x['type'] == 'H']
        l = [x for x in p15 if x['type'] == 'L']

        if len(h) < 2 or len(l) < 2:
            struct_label[i] = 'ND'
            struct_ok[i] = False
            continue

        label = (
            ('HH' if h[-1]['value'] > h[-2]['value'] else 'LH')
            + '+'
            + ('HL' if l[-1]['value'] > l[-2]['value'] else 'LL')
        )
        struct_label[i] = label
        struct_ok[i] = ('HH' in label and 'HL' in label)

    return (
        pd.Series(struct_label, index=df.index, name='Structure'),
        pd.Series(struct_ok, index=df.index, name='Structure_OK'),
    )


def v4_squeeze_flag(df: pd.DataFrame) -> pd.Series:
    """
    2*std20(population) < 1.5*avg20(high-low)
    """
    close = pd.to_numeric(df['Close'], errors='coerce')
    hl = pd.to_numeric(df['High'], errors='coerce') - pd.to_numeric(df['Low'], errors='coerce')
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    atr20 = hl.rolling(20, min_periods=20).mean()
    return ((2.0 * std20) < (1.5 * atr20)).fillna(False)


def v4_true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df['High'], errors='coerce')
    low = pd.to_numeric(df['Low'], errors='coerce')
    prev_close = pd.to_numeric(df['Close'], errors='coerce').shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


# ===========================
# Moteur par Ticker
# ===========================
def _v4_run_ticker(stock_df: pd.DataFrame,
                   idx_close: pd.Series,
                   cfg: dict,
                   idx_sma_on_stock_dates: pd.Series,
                   idx_slope_on_stock_dates: pd.Series):

    stock_df = stock_df.sort_index().copy()

    # --- Indicateurs ---
    rs_line = v4_rs_line(stock_df['Close'], idx_close)
    rsi = v4_rsi(stock_df['Close'], p=14)
    vratio = (stock_df['Volume'] / stock_df['Volume'].rolling(20, min_periods=20).mean()).fillna(0.0)
    mm20 = stock_df['Close'].rolling(20, min_periods=20).mean()
    dist_m20 = ((stock_df['Close'] - mm20).abs() / mm20).fillna(1.0)
    sqz_flag = v4_squeeze_flag(stock_df)

    struct_label, struct_ok = v4_structure_labels(
        stock_df,
        w=cfg['PIVOT_W'],
        last_pivots=cfg['STRUCT_LAST_PIVOTS']
    )

    # --- Filtre prix vs SMA long terme ---
    price_sma_p = int(cfg.get('PRICE_SMA_P', 200))
    price_sma = stock_df['Close'].rolling(price_sma_p, min_periods=price_sma_p).mean()

    price_filter_ok = (
        (stock_df['Close'] >= price_sma).fillna(False)
        if cfg.get('USE_PRICE_SMA_FILTER', False)
        else pd.Series(True, index=stock_df.index)
    )

    # --- SCORE ---
    rs_sma_p = cfg.get('RS_SMA_P', 20)
    rs_sma = rs_line.rolling(window=rs_sma_p).mean()

    if cfg.get('USE_RS_SMA_FILTER', False):
        rs_momentum_ok = (rs_line > rs_sma)
    else:
        rs_momentum_ok = pd.Series(True, index=stock_df.index)

    s_val = pd.Series(0.0, index=stock_df.index)

    # Structure
    s_val += np.where(struct_ok, cfg['W_STRUCT'], 0)

    # Squeeze
    s_val += np.where(sqz_flag, cfg['W_SQZ'], 0)

    # Volume
    s_val += np.where(
        vratio > 1.5,
        cfg['W_VOL'],
        np.where(vratio > 1.1, cfg['W_VOL'] / 2, 0)
    )

    # RSI
    s_val += np.where((rsi >= 50) & (rsi <= 70), cfg['W_RSI'], 0)

    # Distance MM20
    s_val += np.where(
        dist_m20 <= 0.01,
        np.where(
            stock_df['Close'] >= mm20,
            cfg['W_DIST_M20'],
            cfg.get('PENALTY_MM20', -10)
        ),
        0
    )

    # Disqualification RS
    if cfg.get('FORCE_RS_POSITIVE', True):
        mask_final = (rs_line > 0) & rs_momentum_ok
        score = pd.Series(np.where(mask_final, s_val, 0), index=stock_df.index)
    else:
        score = pd.Series(np.where(rs_momentum_ok, s_val, 0), index=stock_df.index)

    # --- Filtre marché ---
    idx_close_on_stock_dates = idx_close.reindex(stock_df.index)

    mkt_ok = (
        (idx_close_on_stock_dates > idx_sma_on_stock_dates.reindex(stock_df.index)).fillna(False)
        if cfg['MKT_FILTER']
        else pd.Series(True, index=stock_df.index)
    )

    # --- ATR ---
    tr = v4_true_range(stock_df)
    atr_vec = tr.rolling(cfg['ATR_P'], min_periods=cfg['ATR_P']).mean().shift(1).fillna(0.0)

    # --- Moteur trading ---
    ledger = []
    active_trade = None

    start_i = max(
        cfg['SMA_P'],
        cfg.get('PRICE_SMA_P', 200) if cfg.get('USE_PRICE_SMA_FILTER', False) else 0
    )

    for date in stock_df.index[start_i:]:
        row = stock_df.loc[date]

        # ======================================================
        # Gestion du trade en cours
        # ======================================================
        if active_trade is not None:
            active_trade['bars_held'] += 1

            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low'] - active_trade['e_px']) / active_trade['e_px']
            c_perf = (row['Close'] - active_trade['e_px']) / active_trade['e_px']

            # --- Logging analytique continu ---
            active_trade['mfe_pct'] = max(active_trade['mfe_pct'], float(h_perf))
            active_trade['mae_pct'] = min(active_trade['mae_pct'], float(l_perf))
            active_trade['max_close_pct'] = max(active_trade['max_close_pct'], float(c_perf))
            active_trade['min_close_pct'] = min(active_trade['min_close_pct'], float(c_perf))

            # BE déclenché sur la barre mais appliqué à partir de la suivante
            be_eligible = active_trade['bars_held'] >= cfg['BE_DELAY']
            be_triggered_this_bar = (
                (not active_trade['be_hit'])
                and (h_perf >= active_trade['be_trig'])
                and be_eligible
            )

            effective_sl = cfg['FEES'] if active_trade['be_hit'] else -cfg['STOP_L']
            hit_tp = (h_perf >= active_trade['tp_val'])
            hit_sl = (l_perf <= effective_sl)

            if hit_tp or hit_sl:
                raw_exit = effective_sl if hit_sl else active_trade['tp_val']
                gain_cash = (raw_exit - cfg['FEES']) * cfg['SIZE']
                trade_type = 'TP' if (hit_tp and not hit_sl) else ('BE' if active_trade['be_hit'] else 'SL')

                if trade_type == 'TP':
                    active_trade['bars_to_tp'] = active_trade['bars_held']
                else:
                    active_trade['bars_to_sl'] = active_trade['bars_held']

                ledger.append({
                    'Ticker': stock_df.attrs.get('Ticker', 'NA'),
                    'Achat': active_trade['date'].strftime('%Y-%m-%d'),
                    'Vente': date.strftime('%Y-%m-%d'),
                    'Gain': float(gain_cash),
                    'Type': trade_type,
                    'Bars': active_trade['bars_held'],

                    # --- Logging analytique ---
                    'MFE_Pct': round(active_trade['mfe_pct'] * 100, 2),
                    'MAE_Pct': round(active_trade['mae_pct'] * 100, 2),
                    'Max_Close_Pct': round(active_trade['max_close_pct'] * 100, 2),
                    'Min_Close_Pct': round(active_trade['min_close_pct'] * 100, 2),
                    'Bars_to_BE': active_trade['bars_to_be'],
                    'Bars_to_TP': active_trade['bars_to_tp'],
                    'Bars_to_SL': active_trade['bars_to_sl'],
                    'TP_Assigned_Pct': round(active_trade['tp_val'] * 100, 2),
                    'BE_Assigned_Pct': round(active_trade['be_trig'] * 100, 2),
                    'BE_Type': active_trade['be_type']
                })
                active_trade = None
            else:
                if be_triggered_this_bar:
                    active_trade['be_hit'] = True
                    if active_trade['bars_to_be'] is None:
                        active_trade['bars_to_be'] = active_trade['bars_held']

            continue

        # ======================================================
        # Entrée
        # ======================================================
        if (
            bool(mkt_ok.loc[date])
            and bool(price_filter_ok.loc[date])
            and float(score.loc[date]) >= cfg['MIN_SCORE']
        ):
            slope = idx_slope_on_stock_dates.reindex(stock_df.index).loc[date]
            vol_pct = float(atr_vec.loc[date] / row['Close']) if row['Close'] != 0 else 0.0

            is_strong = (slope >= cfg['SLOPE_STRONG']) and bool(struct_ok.loc[date])

            current_tp = cfg['TP_TREND']
            if is_strong:
                current_tp += cfg['TP_BOOST']
            if slope < cfg['SLOPE_TRESH']:
                current_tp = cfg['TP_RANGE']

            is_fast_be = (slope >= 0.004 and vol_pct < cfg['VOL_LIM'])
            current_be_trig = cfg['BE_F'] if is_fast_be else cfg['BE_S']

            active_trade = {
                'date': date,
                'e_px': float(row['Close']),
                'tp_val': float(current_tp),
                'be_trig': float(current_be_trig),
                'be_type': 'FAST' if is_fast_be else 'SLOW',
                'be_hit': False,
                'bars_held': 0,

                # --- Logging analytique ---
                'mfe_pct': 0.0,
                'mae_pct': 0.0,
                'max_close_pct': 0.0,
                'min_close_pct': 0.0,
                'bars_to_be': None,
                'bars_to_tp': None,
                'bars_to_sl': None
            }

    # --- Trade en cours ---
    open_trade = None
    if active_trade is not None:
        last_close = float(stock_df.iloc[-1]['Close'])
        perf_actuelle = (last_close - active_trade['e_px']) / active_trade['e_px']

        open_trade = {
            'Ticker': stock_df.attrs.get('Ticker', 'NA'),
            'Date_Achat': active_trade['date'].strftime('%Y-%m-%d'),
            'Prix_Entree': round(active_trade['e_px'], 2),
            'Prix_Actuel': round(last_close, 2),
            'Perf_Latente_Pct': round(perf_actuelle * 100, 2),
            'Objectif_TP_Pct': round(active_trade['tp_val'] * 100, 2),
            'Seuil_BE_Pct': round(active_trade['be_trig'] * 100, 2),
            'Configuration_BE': active_trade['be_type'],
            'Statut_BE': 'SECURISE (BE)' if active_trade['be_hit'] else 'A RISQUE (SL)',
            'Bars_Held': active_trade['bars_held'],

            # --- Logging analytique ---
            'MFE_Pct': round(active_trade['mfe_pct'] * 100, 2),
            'MAE_Pct': round(active_trade['mae_pct'] * 100, 2),
            'Max_Close_Pct': round(active_trade['max_close_pct'] * 100, 2),
            'Min_Close_Pct': round(active_trade['min_close_pct'] * 100, 2),
            'Bars_to_BE': active_trade['bars_to_be']
        }

    df_ledger = pd.DataFrame(ledger)
    stats = {
        'nb_trades': int(len(df_ledger)),
        'gain_total': float(df_ledger['Gain'].sum()) if len(df_ledger) else 0.0,
        'win_rate': float((df_ledger['Gain'] > 0).mean()) if len(df_ledger) else 0.0
    }

    return stats, ledger, open_trade


# ===========================
# alpha4 : moteur multi-tickers
# ===========================
def alpha4(cfg):
    client = bigquery.Client(project=cfg['PROJECT'])
    query = f"SELECT * FROM `{cfg['DB_SET']}.{cfg['TBL']}` ORDER BY Date ASC"
    df = client.query(query).to_dataframe()

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    for c in ['Close', 'High', 'Low', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    idx_ticker = cfg['IDX']
    base_idx = df[df['Ticker'] == idx_ticker].copy().set_index('Date').sort_index()
    idx_close = base_idx['Close']

    idx_sma = idx_close.rolling(cfg['SMA_P'], min_periods=cfg['SMA_P']).mean()
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4)).fillna(0)

    universe = [t for t in sorted(df['Ticker'].dropna().unique()) if t != idx_ticker]
    if cfg['UNIVERSE']:
        universe = [t for t in universe if t in cfg['UNIVERSE']]

    portfolio_trades = []
    portfolio_open_positions = []
    per_ticker_stats = {}

    for t in universe:
        d = df[df['Ticker'] == t].copy().set_index('Date').sort_index()
        d.attrs['Ticker'] = t

        if len(d) < 100:
            continue

        stats, trades, open_trade = _v4_run_ticker(
            d,
            idx_close,
            cfg,
            idx_sma.reindex(d.index),
            idx_slope.reindex(d.index)
        )

        per_ticker_stats[t] = stats
        portfolio_trades.extend(trades)

        if open_trade:
            portfolio_open_positions.append(open_trade)

    df_ledger = pd.DataFrame(portfolio_trades)

    return {
        'metadata': {
            'system': 'Titanium v4.2 + logging',
            'universe': len(universe)
        },
        'portfolio': {
            'gain_total': float(df_ledger['Gain'].sum()) if len(df_ledger) else 0.0,
            'nb_trades': int(len(df_ledger)) if len(df_ledger) else 0,
            'win_rate': float((df_ledger['Gain'] > 0).mean()) if len(df_ledger) else 0.0,
            'by_ticker': per_ticker_stats
        },
        'open_positions': portfolio_open_positions,
        'trades': df_ledger.to_dict(orient='records') if len(df_ledger) else []
    }


if __name__ == '__main__':
    out = alpha4(ALPHA4_CFG)
    print(json.dumps(out, indent=2, ensure_ascii=False))
# ==== END sigma2.py
