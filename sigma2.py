# ==== START sigma2.py
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery

ALPHA4_CFG = {
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours',
    'IDX': '^FCHI',          # Indice de marché (CAC 40)
    'MKT_FILTER': True,      # Filtre marché : close > SMA100 CAC
    'SMA_P': 100,            # SMA marché
    'MIN_SCORE': 86,         # Seuil de score
    'LOOKBACK': 63,          # Fenêtre RS (=> current vs current-62)
    'TP_TREND': 0.13,
    'TP_RANGE': 0.10,
    'SLOPE_TRESH': -0.003,   # seuil pente SMA marché
    'ATR_P': 50,
    'BE_F': 0.06,            # BE fast
    'BE_S': 0.0495,          # BE slow
    'VOL_LIM': 0.025,        # limite vol pour BE fast
    'STOP_L': 0.10,
    'FEES': 0.0056,
    'SIZE': 4000,
    'PIVOT_W': 3,
    'STRUCT_LAST_PIVOTS': 15,
    # Optionnel: restreindre l’univers, ex. ['EN.PA','ORA.PA'] ; None => tous
    'UNIVERSE': None,
}

# ===========================
#  Indicateurs (parité GAS) — NOMS DIFFERENTS
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
    idx_close   = pd.to_numeric(idx_close,   errors='coerce').sort_index()

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

        s_curr = s_vals[i];     s_prev = s_vals[i - 62]
        i_curr = i_vals[pos];   i_prev = i_vals[pos - 62]

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
    close  = pd.to_numeric(close, errors='coerce')
    diff   = close.diff()
    gains  = diff.clip(lower=0).rolling(p, min_periods=p).sum()
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
            if lows[j]  <= lows[i]:
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
    n  = len(df)
    piv = v4_pivot_events(df, w=w)

    visible_on = [[] for _ in range(n)]
    for p in piv:
        vis_i = p['pivot_i'] + w
        if vis_i < n:
            visible_on[vis_i].append(p)

    active = []
    struct_label = np.array(['ND'] * n, dtype=object)
    struct_ok    = np.zeros(n, dtype=bool)

    for i in range(n):
        if visible_on[i]:
            active.extend(visible_on[i])

        p15 = active[-last_pivots:]
        h = [x for x in p15 if x['type'] == 'H']
        l = [x for x in p15 if x['type'] == 'L']

        if len(h) < 2 or len(l) < 2:
            struct_label[i] = 'ND'
            struct_ok[i]    = False
            continue

        label = (
            ('HH' if h[-1]['value'] > h[-2]['value'] else 'LH')
            + '+'
            + ('HL' if l[-1]['value'] > l[-2]['value'] else 'LL')
        )
        struct_label[i] = label
        struct_ok[i]    = ('HH' in label and 'HL' in label)

    return (
        pd.Series(struct_label, index=df.index, name='Structure'),
        pd.Series(struct_ok,    index=df.index, name='Structure_OK'),
    )


def v4_squeeze_flag(df: pd.DataFrame) -> pd.Series:
    """
    2*std20(population) < 1.5*avg20(high-low)
    """
    close = pd.to_numeric(df['Close'], errors='coerce')
    hl    = pd.to_numeric(df['High'],  errors='coerce') - pd.to_numeric(df['Low'], errors='coerce')
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    atr20 = hl.rolling(20,   min_periods=20).mean()
    return ((2.0 * std20) < (1.5 * atr20)).fillna(False)


def v4_true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df['High'], errors='coerce')
    low  = pd.to_numeric(df['Low'],  errors='coerce')
    prev_close = pd.to_numeric(df['Close'], errors='coerce').shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


# ===========================
#  Moteur par Ticker — NOM DIFFERENT
# ===========================
def _v4_run_ticker(stock_df: pd.DataFrame,
                   idx_close: pd.Series,
                   cfg: dict,
                   idx_sma_on_stock_dates: pd.Series,
                   idx_slope_on_stock_dates: pd.Series):
    """
    Exécute le moteur (score/entrées/BE/TP/SL) pour un ticker donné.
    Retourne: stats, ledger(trades)
    """
    stock_df = stock_df.sort_index().copy()

    # Indicateurs
    rs_line  = v4_rs_line(stock_df['Close'], idx_close)
    rsi      = v4_rsi(stock_df['Close'], p=14)
    vratio   = (stock_df['Volume'] / stock_df['Volume'].rolling(20, min_periods=20).mean()).fillna(0.0)
    mm20     = stock_df['Close'].rolling(20, min_periods=20).mean()
    dist_m20 = ((stock_df['Close'] - mm20).abs() / mm20).fillna(1.0)
    sqz_flag = v4_squeeze_flag(stock_df)
    struct_label, struct_ok = v4_structure_labels(stock_df, w=cfg['PIVOT_W'], last_pivots=cfg['STRUCT_LAST_PIVOTS'])

    # Score
    s_val = pd.Series(0.0, index=stock_df.index)
    s_val += np.where((rsi >= 50) & (rsi <= 70), 15, 0)
    s_val += np.where(vratio > 1.5, 25, np.where(vratio > 1.1, 12.5, 0))
    s_val += np.where(dist_m20 <= 0.01, np.where(stock_df['Close'] >= mm20, 20, -10), 0)
    s_val += np.where(struct_ok, 30, 0)
    s_val += np.where(sqz_flag, 10, 0)
    score = pd.Series(np.where(rs_line <= 0, 0, s_val), index=stock_df.index)

    # Filtre marché & ATR
    idx_close_on_stock_dates = idx_close.reindex(stock_df.index)
    if cfg['MKT_FILTER']:
        mkt_ok = (idx_close_on_stock_dates > idx_sma_on_stock_dates.reindex(stock_df.index)).fillna(False)
    else:
        mkt_ok = pd.Series(True, index=stock_df.index)

    tr = v4_true_range(stock_df)
    atr_vec = tr.rolling(cfg['ATR_P'], min_periods=cfg['ATR_P']).mean().shift(1).fillna(0.0)

    # Moteur trading
    ledger = []
    active_trade = None
    start_i = cfg['SMA_P']  # on attend au moins SMA_P barres

    for date in stock_df.index[start_i:]:
        row = stock_df.loc[date]

        if active_trade is not None:
            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low']  - active_trade['e_px']) / active_trade['e_px']

            # BE constaté sur la barre, appliqué seulement après l'éval TP/SL
            be_triggered_this_bar = (not active_trade['be_hit']) and (h_perf >= active_trade['be_trig'])

            effective_sl = cfg['FEES'] if active_trade['be_hit'] else -cfg['STOP_L']
            hit_tp = (h_perf >= active_trade['tp_val'])
            hit_sl = (l_perf <= effective_sl)

            if hit_tp or hit_sl:
                # Règle pessimiste: TP & SL touchés => on retient SL
                raw_exit = effective_sl if hit_sl else active_trade['tp_val']
                gain_cash = (raw_exit - cfg['FEES']) * cfg['SIZE']
                trade_type = 'TP' if (hit_tp and not hit_sl) else ('BE' if active_trade['be_hit'] else 'SL')
                ledger.append({
                    'Ticker': stock_df.attrs.get('Ticker', 'NA'),
                    'Achat': active_trade['date'].strftime('%Y-%m-%d'),
                    'Vente': date.strftime('%Y-%m-%d'),
                    'Gain': float(gain_cash),
                    'Type': trade_type
                })
                active_trade = None
            else:
                if be_triggered_this_bar:
                    active_trade['be_hit'] = True
            continue

        # Entrée
        if bool(mkt_ok.loc[date]) and float(score.loc[date]) >= cfg['MIN_SCORE']:
            vol_pct = float(atr_vec.loc[date] / row['Close']) if pd.notna(row['Close']) and row['Close'] != 0 else 0.0
            slope   = idx_slope_on_stock_dates.reindex(stock_df.index).loc[date]
            tp_val  = cfg['TP_TREND'] if slope >= cfg['SLOPE_TRESH'] else cfg['TP_RANGE']
            be_trig = cfg['BE_F'] if (slope >= 0.004 and vol_pct < cfg['VOL_LIM']) else cfg['BE_S']

            active_trade = {
                'date': date,
                'e_px': float(row['Close']),  # pour next-bar open: remplacer ici
                'tp_val': tp_val,
                'be_trig': be_trig,
                'be_hit': False
            }

    # Stats
    df_ledger = pd.DataFrame(ledger)
    nb_trades = int(len(df_ledger))
    gain_total = float(df_ledger['Gain'].sum()) if nb_trades else 0.0
    win_rate = float((df_ledger['Gain'] > 0).mean()) if nb_trades else 0.0

    stats = {
        'nb_trades': nb_trades,
        'gain_total': gain_total,
        'win_rate': win_rate
    }
    return stats, ledger


# ===========================
#  alpha4 : moteur multi-tickers — NOM DIFFERENT
# ===========================
def alpha4():
    """
    Exécute le moteur sur l'ensemble des tickers de CC_Historique_Cours (hors indice),
    et renvoie un JSON avec :
      - metadata (système, univers, index),
      - portfolio (stats consolidées + stats par ticker),
      - trades (tous les trades avec leur ticker).
    """
    cfg = ALPHA4_CFG
    client = bigquery.Client(project=cfg['PROJECT'])

    query = f"""
        SELECT *
        FROM `{cfg['DB_SET']}.{cfg['TBL']}`
        ORDER BY Date ASC
    """
    df = client.query(query).to_dataframe()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Nettoyage
    for c in ['Close','High','Low','Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Séparation indice / titres
    idx_ticker = cfg['IDX']
    base_idx = df[df['Ticker'] == idx_ticker].copy()
    if base_idx.empty:
        raise RuntimeError(f"Indice {idx_ticker} introuvable dans la table.")

    base_idx = base_idx.set_index('Date').sort_index()
    idx_close = base_idx['Close']

    # SMA marché + pente sur 4 barres (slope ~ GAS)
    idx_sma   = idx_close.rolling(cfg['SMA_P'], min_periods=cfg['SMA_P']).mean()
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4)).fillna(0)

    # Univers de tickers
    all_tickers = sorted(df['Ticker'].dropna().unique().tolist())
    universe = [t for t in all_tickers if t != idx_ticker]
    if cfg['UNIVERSE'] is not None:
        allowed = set(cfg['UNIVERSE'])
        universe = [t for t in universe if t in allowed]

    portfolio_trades = []
    per_ticker_stats = {}

    for t in universe:
        d = df[df['Ticker'] == t].copy()
        if d.empty:
            per_ticker_stats[t] = {'nb_trades': 0, 'gain_total': 0.0, 'win_rate': 0.0}
            continue

        d = d.set_index('Date').sort_index()
        d.attrs['Ticker'] = t

        # ignore séries trop courtes (< 100 barres)
        if len(d) < 100:
            per_ticker_stats[t] = {'nb_trades': 0, 'gain_total': 0.0, 'win_rate': 0.0}
            continue

        # Projeter marché sur dates du titre
        idx_sma_on_stock_dates   = idx_sma.reindex(d.index)
        idx_slope_on_stock_dates = idx_slope.reindex(d.index)

        stats, trades = _v4_run_ticker(
            stock_df=d,
            idx_close=idx_close,
            cfg=cfg,
            idx_sma_on_stock_dates=idx_sma_on_stock_dates,
            idx_slope_on_stock_dates=idx_slope_on_stock_dates
        )
        per_ticker_stats[t] = stats
        portfolio_trades.extend(trades)

    # Consolidation portefeuille
    df_ledger = pd.DataFrame(portfolio_trades)
    nb_trades  = int(len(df_ledger))
    gain_total = float(df_ledger['Gain'].sum()) if nb_trades else 0.0
    win_rate   = float((df_ledger['Gain'] > 0).mean()) if nb_trades else 0.0

    res = {
        'metadata': {
            'system': 'Alpha Engine v4 (GAS-parity)',
            'index': idx_ticker,
            'universe_size': len(universe)
        },
        'portfolio': {
            'gain_total': gain_total,
            'nb_trades': nb_trades,
            'win_rate': win_rate,
            'by_ticker': per_ticker_stats
        },
        'trades': df_ledger.to_dict(orient='records') if nb_trades else []
    }
    return res


if __name__ == '__main__':
    out = alpha4()
    print(json.dumps(out, indent=2, ensure_ascii=False))
# ==== END sigma2.py
