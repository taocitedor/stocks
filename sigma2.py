# ==== START sigma2.py ====
import json
import copy
import numpy as np
import pandas as pd
from google.cloud import bigquery

# =========================================================
# CONFIG DE BASE
# =========================================================
ALPHA4_CFG = {
    # --- Configuration Backend ---
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours_v2',
    'IDX': '^FCHI',

    # --- Moteur principal ---
    'MKT_FILTER': True,
    'SMA_P': 100,
    'LOOKBACK': 63,

    # --- Cœur de la stratégie ---
    'MIN_SCORE': 86,

    # --- Pondérations score ---
    'W_STRUCT': 30,
    'W_VOL': 25,
    'W_DIST_M20': 20,
    'W_RSI': 15,
    'W_SQZ': 10,

    # --- Pénalité / disqualification ---
    'PENALTY_MM20': -10,
    'FORCE_RS_POSITIVE': True,

    # --- Filtre vélocité ---
    'USE_RS_SMA_FILTER': False,
    'RS_SMA_P': 20,

    # --- Filtre tendance titre ---
    'USE_PRICE_SMA_FILTER': True,
    'PRICE_SMA_P': 200,

    # --- TP dynamiques ---
    'TP_TREND': 0.135,
    'TP_RANGE': 0.10,
    'TP_BOOST': 0.025,
    'SLOPE_TRESH': 0.002,
    'SLOPE_STRONG': 0.007,

    # --- BE adaptatif ---
    'BE_F': 0.06,
    'BE_S': 0.0495,
    'BE_DELAY': 3,
    'VOL_LIM': 0.025,

    # --- Profit Lock ---
    'USE_PROFIT_LOCK': True,
    'LOCK1_TRIGGER': 0.095,
    'LOCK1_RAW': 0.0206,
    'LOCK2_TRIGGER': 999.0,
    'LOCK2_RAW': 0.0356,

    # --- Gestion risque & frais ---
    'ATR_P': 50,
    'STOP_L': 0.10,
    'FEES': 0.0056,
    'SIZE': 4000,

    # --- Structure & pivots ---
    'PIVOT_W': 3,
    'STRUCT_LAST_PIVOTS': 15,

    # --- Filtre régime déjà validé ---
    'EXCLUDE_TP135_SLOW': True,

    # --- Filtre 10 + SLOW walk-forward ---
    # Laisser False ici : le walk-forward active les variantes lui-même.
    'USE_RANGE_SLOW_CONTEXT_FILTER': False,
    'RANGE_SLOW_SCORE_MODE': 'EQ100',   # 'EQ100' ou 'GE97_5'
    'RANGE_SLOW_MAX_IDX_GAP': 1.5,      # 1.0 / 1.5 / 2.0

    # --- Univers ---
    'UNIVERSE': None
}

# =========================================================
# VARIANTES A TESTER (fixées d'avance)
# =========================================================
VARIANTS = [
    {'name': 'EQ100_GAP1.0',  'RANGE_SLOW_SCORE_MODE': 'EQ100',  'RANGE_SLOW_MAX_IDX_GAP': 1.0},
    {'name': 'EQ100_GAP1.5',  'RANGE_SLOW_SCORE_MODE': 'EQ100',  'RANGE_SLOW_MAX_IDX_GAP': 1.5},
    {'name': 'EQ100_GAP2.0',  'RANGE_SLOW_SCORE_MODE': 'EQ100',  'RANGE_SLOW_MAX_IDX_GAP': 2.0},
    {'name': 'GE97_5_GAP1.0', 'RANGE_SLOW_SCORE_MODE': 'GE97_5', 'RANGE_SLOW_MAX_IDX_GAP': 1.0},
    {'name': 'GE97_5_GAP1.5', 'RANGE_SLOW_SCORE_MODE': 'GE97_5', 'RANGE_SLOW_MAX_IDX_GAP': 1.5},
    {'name': 'GE97_5_GAP2.0', 'RANGE_SLOW_SCORE_MODE': 'GE97_5', 'RANGE_SLOW_MAX_IDX_GAP': 2.0},
]

# =========================================================
# FOLDS WALK-FORWARD
# =========================================================
WALKFORWARD_FOLDS = [
    {'name': 'WF_1', 'train_start': 2019, 'train_end': 2022, 'test_year': 2023},
    {'name': 'WF_2', 'train_start': 2019, 'train_end': 2023, 'test_year': 2024},
    {'name': 'WF_3', 'train_start': 2019, 'train_end': 2024, 'test_year': 2025},
]

LINE_SIZE = 4000.0


# =========================================================
# CHARGEMENT DES DONNEES (UNE SEULE FOIS)
# =========================================================
def load_market_data(cfg):
    client = bigquery.Client(project=cfg['PROJECT'])
    query = f"SELECT * FROM `{cfg['DB_SET']}.{cfg['TBL']}` ORDER BY Date ASC"
    df = client.query(query).to_dataframe()

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    for c in ['Close', 'High', 'Low', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


# =========================================================
# INDICATEURS
# =========================================================
def v4_rs_line(stock_close: pd.Series, idx_close: pd.Series) -> pd.Series:
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


def v4_rsi(close: pd.Series, p: int = 14) -> pd.Series:
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
    highs = pd.to_numeric(df['High'], errors='coerce').to_numpy(dtype=float)
    lows = pd.to_numeric(df['Low'], errors='coerce').to_numpy(dtype=float)

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

        p_last = active[-last_pivots:]
        h = [x for x in p_last if x['type'] == 'H']
        l = [x for x in p_last if x['type'] == 'L']

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


# =========================================================
# BACKTEST PAR TICKER
# =========================================================
def _v4_run_ticker(stock_df: pd.DataFrame,
                   idx_close: pd.Series,
                   cfg: dict,
                   idx_sma_on_stock_dates: pd.Series,
                   idx_slope_on_stock_dates: pd.Series):

    stock_df = stock_df.sort_index().copy()

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

    price_sma_p = int(cfg.get('PRICE_SMA_P', 200))
    price_sma = stock_df['Close'].rolling(price_sma_p, min_periods=price_sma_p).mean()
    price_filter_ok = (
        (stock_df['Close'] >= price_sma).fillna(False)
        if cfg.get('USE_PRICE_SMA_FILTER', False)
        else pd.Series(True, index=stock_df.index)
    )

    rs_sma_p = int(cfg.get('RS_SMA_P', 20))
    rs_sma = rs_line.rolling(window=rs_sma_p).mean()
    if cfg.get('USE_RS_SMA_FILTER', False):
        rs_momentum_ok = (rs_line > rs_sma)
    else:
        rs_momentum_ok = pd.Series(True, index=stock_df.index)

    s_val = pd.Series(0.0, index=stock_df.index)
    s_val += np.where(struct_ok, cfg['W_STRUCT'], 0)
    s_val += np.where(sqz_flag, cfg['W_SQZ'], 0)
    s_val += np.where(
        vratio > 1.5,
        cfg['W_VOL'],
        np.where(vratio > 1.1, cfg['W_VOL'] / 2, 0)
    )
    s_val += np.where((rsi >= 50) & (rsi <= 70), cfg['W_RSI'], 0)
    s_val += np.where(
        dist_m20 <= 0.01,
        np.where(stock_df['Close'] >= mm20, cfg['W_DIST_M20'], cfg.get('PENALTY_MM20', -10)),
        0
    )

    if cfg.get('FORCE_RS_POSITIVE', True):
        mask_final = (rs_line > 0) & rs_momentum_ok
        score = pd.Series(np.where(mask_final, s_val, 0), index=stock_df.index)
    else:
        score = pd.Series(np.where(rs_momentum_ok, s_val, 0), index=stock_df.index)

    idx_close_on_stock_dates = idx_close.reindex(stock_df.index)
    mkt_ok = (
        (idx_close_on_stock_dates > idx_sma_on_stock_dates.reindex(stock_df.index)).fillna(False)
        if cfg['MKT_FILTER']
        else pd.Series(True, index=stock_df.index)
    )

    tr = v4_true_range(stock_df)
    atr_vec = tr.rolling(cfg['ATR_P'], min_periods=cfg['ATR_P']).mean().shift(1).fillna(0.0)

    ledger = []
    active_trade = None

    start_i = max(
        cfg['SMA_P'],
        cfg.get('PRICE_SMA_P', 200) if cfg.get('USE_PRICE_SMA_FILTER', False) else 0
    )

    skipped_tp135_slow = 0
    skipped_range_slow_context = 0

    for date in stock_df.index[start_i:]:
        row = stock_df.loc[date]

        # -----------------------------------------------------
        # GESTION DU TRADE EN COURS
        # -----------------------------------------------------
        if active_trade is not None:
            active_trade['bars_held'] += 1

            h_perf = (row['High'] - active_trade['e_px']) / active_trade['e_px']
            l_perf = (row['Low'] - active_trade['e_px']) / active_trade['e_px']
            c_perf = (row['Close'] - active_trade['e_px']) / active_trade['e_px']

            active_trade['mfe_pct'] = max(active_trade['mfe_pct'], float(h_perf))
            active_trade['mae_pct'] = min(active_trade['mae_pct'], float(l_perf))
            active_trade['max_close_pct'] = max(active_trade['max_close_pct'], float(c_perf))
            active_trade['min_close_pct'] = min(active_trade['min_close_pct'], float(c_perf))

            be_eligible = active_trade['bars_held'] >= cfg['BE_DELAY']
            be_triggered_this_bar = (
                (not active_trade['be_hit'])
                and (h_perf >= active_trade['be_trig'])
                and be_eligible
            )

            if cfg.get('USE_PROFIT_LOCK', False):
                if active_trade['mfe_pct'] >= cfg.get('LOCK2_TRIGGER', 0.10):
                    active_trade['profit_lock_raw'] = cfg.get('LOCK2_RAW', 0.0356)
                    active_trade['profit_lock_level'] = 'LOCK2'
                elif active_trade['mfe_pct'] >= cfg.get('LOCK1_TRIGGER', 0.095):
                    if active_trade['profit_lock_raw'] is None:
                        active_trade['profit_lock_raw'] = cfg.get('LOCK1_RAW', 0.0206)
                        active_trade['profit_lock_level'] = 'LOCK1'

            effective_sl = cfg['FEES'] if active_trade['be_hit'] else -cfg['STOP_L']
            if active_trade.get('profit_lock_raw') is not None:
                effective_sl = max(effective_sl, active_trade['profit_lock_raw'])

            hit_tp = (h_perf >= active_trade['tp_val'])
            hit_sl = (l_perf <= effective_sl)

            if hit_tp or hit_sl:
                raw_exit = effective_sl if hit_sl else active_trade['tp_val']
                gain_cash = (raw_exit - cfg['FEES']) * cfg['SIZE']

                if hit_tp and not hit_sl:
                    trade_type = 'TP'
                else:
                    if active_trade.get('profit_lock_raw') is not None and effective_sl > cfg['FEES']:
                        trade_type = active_trade.get('profit_lock_level', 'LOCK')
                    else:
                        trade_type = 'BE' if active_trade['be_hit'] else 'SL'

                if trade_type == 'TP':
                    active_trade['bars_to_tp'] = active_trade['bars_held']
                else:
                    active_trade['bars_to_sl'] = active_trade['bars_held']

                idx_exit_px = idx_close.reindex(stock_df.index).loc[date]

                idx_return_trade_pct = None
                excess_return_vs_idx_pct = None
                stock_return_trade_pct = (raw_exit - cfg['FEES']) * 100.0

                if (
                    active_trade.get('idx_entry_px') is not None
                    and pd.notna(idx_exit_px)
                    and active_trade['idx_entry_px'] != 0
                ):
                    idx_return_trade_pct = ((float(idx_exit_px) / float(active_trade['idx_entry_px'])) - 1.0) * 100.0
                    excess_return_vs_idx_pct = stock_return_trade_pct - idx_return_trade_pct

                ledger.append({
                    'Ticker': stock_df.attrs.get('Ticker', 'NA'),
                    'Achat': active_trade['date'].strftime('%Y-%m-%d'),
                    'Vente': date.strftime('%Y-%m-%d'),
                    'Gain': float(gain_cash),
                    'Type': trade_type,
                    'Bars': active_trade['bars_held'],

                    'MFE_Pct': round(active_trade['mfe_pct'] * 100, 2),
                    'MAE_Pct': round(active_trade['mae_pct'] * 100, 2),
                    'Max_Close_Pct': round(active_trade['max_close_pct'] * 100, 2),
                    'Min_Close_Pct': round(active_trade['min_close_pct'] * 100, 2),
                    'Bars_to_BE': active_trade['bars_to_be'],
                    'Bars_to_TP': active_trade['bars_to_tp'],
                    'Bars_to_SL': active_trade['bars_to_sl'],
                    'TP_Assigned_Pct': round(active_trade['tp_val'] * 100, 2),
                    'BE_Assigned_Pct': round(active_trade['be_trig'] * 100, 2),
                    'BE_Type': active_trade['be_type'],

                    'Profit_Lock_Level': active_trade.get('profit_lock_level', None),
                    'Profit_Lock_Raw_Pct': round(active_trade['profit_lock_raw'] * 100, 2)
                    if active_trade.get('profit_lock_raw') is not None else None,

                    # contexte d'entrée
                    'Score_Entry': round(active_trade['score_entry'], 2),
                    'RS_Line_Entry': round(active_trade['rs_line_entry'], 2),
                    'RS_SMA_Entry': round(active_trade['rs_sma_entry'], 2)
                    if active_trade['rs_sma_entry'] is not None else None,
                    'RS_Momentum_OK_Entry': active_trade['rs_momentum_ok_entry'],
                    'RSI_Entry': round(active_trade['rsi_entry'], 2),
                    'Volume_Ratio_Entry': round(active_trade['volume_ratio_entry'], 3),
                    'Dist_M20_Entry_Pct': round(active_trade['dist_m20_entry_pct'], 2),
                    'Squeeze_Flag_Entry': active_trade['squeeze_flag_entry'],
                    'Structure_Label_Entry': active_trade['structure_label_entry'],
                    'Structure_OK_Entry': active_trade['structure_ok_entry'],
                    'Price_Filter_OK_Entry': active_trade['price_filter_ok_entry'],
                    'Price_vs_SMA200_Entry_Pct': round(active_trade['price_vs_sma200_entry_pct'], 2)
                    if active_trade['price_vs_sma200_entry_pct'] is not None else None,

                    'Idx_Close_Entry': round(active_trade['idx_entry_px'], 2)
                    if active_trade['idx_entry_px'] is not None else None,
                    'Idx_SMA_Entry': round(active_trade['idx_entry_sma'], 2)
                    if active_trade['idx_entry_sma'] is not None else None,
                    'Idx_Gap_vs_SMA_Entry_Pct': round(active_trade['idx_gap_vs_sma_entry_pct'], 3)
                    if active_trade['idx_gap_vs_sma_entry_pct'] is not None else None,
                    'Idx_Slope_Entry_Pct': round(active_trade['idx_slope_entry_pct'], 3)
                    if active_trade['idx_slope_entry_pct'] is not None else None,
                    'Mkt_Filter_OK_Entry': active_trade['mkt_filter_ok_entry'],

                    'Vol_Pct_Entry': round(active_trade['vol_pct_entry'], 3),
                    'Is_FAST_BE_Entry': active_trade['is_fast_be_entry'],
                    'Is_Strong_Trend_Entry': active_trade['is_strong_trend_entry'],
                    'TP_Regime_Source': active_trade['tp_regime_source'],

                    'Idx_Close_Exit': round(float(idx_exit_px), 2) if pd.notna(idx_exit_px) else None,
                    'Stock_Return_Trade_Pct': round(stock_return_trade_pct, 2),
                    'Idx_Return_Trade_Pct': round(idx_return_trade_pct, 2) if idx_return_trade_pct is not None else None,
                    'Excess_Return_vs_Idx_Pct': round(excess_return_vs_idx_pct, 2) if excess_return_vs_idx_pct is not None else None
                })

                active_trade = None
            else:
                if be_triggered_this_bar:
                    active_trade['be_hit'] = True
                    if active_trade['bars_to_be'] is None:
                        active_trade['bars_to_be'] = active_trade['bars_held']
            continue

        # -----------------------------------------------------
        # ENTREE
        # -----------------------------------------------------
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

            idx_entry_px = idx_close.reindex(stock_df.index).loc[date]
            idx_entry_sma = idx_sma_on_stock_dates.reindex(stock_df.index).loc[date]

            idx_gap_vs_sma_pct = None
            if pd.notna(idx_entry_px) and pd.notna(idx_entry_sma) and idx_entry_sma != 0:
                idx_gap_vs_sma_pct = ((float(idx_entry_px) / float(idx_entry_sma)) - 1.0) * 100.0

            # filtre existant : exclure TP13.5 + SLOW
            if cfg.get('EXCLUDE_TP135_SLOW', False):
                is_tp_135 = abs(current_tp - cfg['TP_TREND']) < 1e-12
                is_slow = not is_fast_be
                if is_tp_135 and is_slow:
                    skipped_tp135_slow += 1
                    continue

            # NOUVEAU filtre walk-forward sur 10 + SLOW
            is_range_slow = (abs(current_tp - cfg['TP_RANGE']) < 1e-12) and (not is_fast_be)
            if cfg.get('USE_RANGE_SLOW_CONTEXT_FILTER', False) and is_range_slow:
                score_now = float(score.loc[date])

                score_condition = False
                if cfg.get('RANGE_SLOW_SCORE_MODE') == 'EQ100':
                    score_condition = (abs(score_now - 100.0) < 1e-12)
                elif cfg.get('RANGE_SLOW_SCORE_MODE') == 'GE97_5':
                    score_condition = (score_now >= 97.5)
                else:
                    raise ValueError(f"RANGE_SLOW_SCORE_MODE inconnu: {cfg.get('RANGE_SLOW_SCORE_MODE')}")

                if (
                    score_condition
                    and idx_gap_vs_sma_pct is not None
                    and idx_gap_vs_sma_pct <= cfg['RANGE_SLOW_MAX_IDX_GAP']
                ):
                    skipped_range_slow_context += 1
                    continue

            tp_regime_source = 'TREND'
            if is_strong:
                tp_regime_source = 'TREND_BOOST'
            if slope < cfg['SLOPE_TRESH']:
                tp_regime_source = 'RANGE'

            price_sma_entry = price_sma.loc[date] if date in price_sma.index else np.nan
            price_vs_sma200_pct = None
            if pd.notna(price_sma_entry) and price_sma_entry != 0:
                price_vs_sma200_pct = ((float(row['Close']) / float(price_sma_entry)) - 1.0) * 100.0

            rs_sma_entry = rs_sma.loc[date] if date in rs_sma.index else np.nan

            active_trade = {
                'date': date,
                'e_px': float(row['Close']),
                'tp_val': float(current_tp),
                'be_trig': float(current_be_trig),
                'be_type': 'FAST' if is_fast_be else 'SLOW',
                'be_hit': False,
                'bars_held': 0,

                'profit_lock_raw': None,
                'profit_lock_level': None,

                'mfe_pct': 0.0,
                'mae_pct': 0.0,
                'max_close_pct': 0.0,
                'min_close_pct': 0.0,
                'bars_to_be': None,
                'bars_to_tp': None,
                'bars_to_sl': None,

                # contexte d'entrée
                'score_entry': float(score.loc[date]),
                'rs_line_entry': float(rs_line.loc[date]),
                'rs_sma_entry': float(rs_sma_entry) if pd.notna(rs_sma_entry) else None,
                'rs_momentum_ok_entry': bool(rs_momentum_ok.loc[date]),
                'rsi_entry': float(rsi.loc[date]),
                'volume_ratio_entry': float(vratio.loc[date]),
                'dist_m20_entry_pct': float(dist_m20.loc[date] * 100.0),
                'squeeze_flag_entry': bool(sqz_flag.loc[date]),
                'structure_label_entry': struct_label.loc[date],
                'structure_ok_entry': bool(struct_ok.loc[date]),
                'price_filter_ok_entry': bool(price_filter_ok.loc[date]),
                'price_vs_sma200_entry_pct': float(price_vs_sma200_pct) if price_vs_sma200_pct is not None else None,

                'idx_entry_px': float(idx_entry_px) if pd.notna(idx_entry_px) else None,
                'idx_entry_sma': float(idx_entry_sma) if pd.notna(idx_entry_sma) else None,
                'idx_gap_vs_sma_entry_pct': float(idx_gap_vs_sma_pct) if idx_gap_vs_sma_pct is not None else None,
                'idx_slope_entry_pct': float(slope * 100.0) if pd.notna(slope) else None,
                'mkt_filter_ok_entry': bool(mkt_ok.loc[date]),

                'vol_pct_entry': float(vol_pct * 100.0),
                'is_fast_be_entry': bool(is_fast_be),
                'is_strong_trend_entry': bool(is_strong),
                'tp_regime_source': tp_regime_source
            }

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
            'Bars_Held': active_trade['bars_held']
        }

    df_ledger = pd.DataFrame(ledger)
    stats = {
        'nb_trades': int(len(df_ledger)),
        'gain_total': float(df_ledger['Gain'].sum()) if len(df_ledger) else 0.0,
        'win_rate': float((df_ledger['Gain'] > 0).mean()) if len(df_ledger) else 0.0,
        'skipped_tp135_slow': int(skipped_tp135_slow),
        'skipped_range_slow_context': int(skipped_range_slow_context)
    }
    return stats, ledger, open_trade


# =========================================================
# BACKTEST GLOBAL
# =========================================================
def alpha4(cfg, base_df=None):
    if base_df is None:
        df = load_market_data(cfg)
    else:
        df = base_df.copy()

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
    total_skipped_tp135_slow = int(sum(v.get('skipped_tp135_slow', 0) for v in per_ticker_stats.values()))
    total_skipped_range_slow_context = int(sum(v.get('skipped_range_slow_context', 0) for v in per_ticker_stats.values()))

    return {
        'metadata': {
            'system': 'Titanium walk-forward test harness cached',
            'universe': len(universe),
            'exclude_tp135_slow': cfg.get('EXCLUDE_TP135_SLOW', False),
            'use_range_slow_context_filter': cfg.get('USE_RANGE_SLOW_CONTEXT_FILTER', False),
            'range_slow_score_mode': cfg.get('RANGE_SLOW_SCORE_MODE'),
            'range_slow_max_idx_gap': cfg.get('RANGE_SLOW_MAX_IDX_GAP'),
            'total_skipped_tp135_slow': total_skipped_tp135_slow,
            'total_skipped_range_slow_context': total_skipped_range_slow_context
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


# =========================================================
# METRIQUES
# =========================================================
def compute_xirr_from_ledger(df_ledger: pd.DataFrame) -> float:
    if df_ledger is None or len(df_ledger) == 0:
        return np.nan

    flows = []
    for _, r in df_ledger.iterrows():
        flows.append((pd.Timestamp(r['Achat']).normalize(), -LINE_SIZE))
        flows.append((pd.Timestamp(r['Vente']).normalize(), LINE_SIZE + float(r['Gain'])))

    cf = pd.DataFrame(flows, columns=['Date', 'CashFlow'])
    cf = cf.groupby('Date', as_index=False)['CashFlow'].sum().sort_values('Date')

    if len(cf) < 2:
        return np.nan

    base = cf['Date'].min()
    am = cf['CashFlow'].tolist()
    dt = cf['Date'].tolist()

    def xnpv(rate):
        return sum(a / ((1 + rate) ** (((d - base).days / 365.25))) for a, d in zip(am, dt))

    lo, hi = -0.9999, 0.1
    vlo, vhi = xnpv(lo), xnpv(hi)

    while np.isfinite(vlo) and np.isfinite(vhi) and vlo * vhi > 0 and hi < 1e6:
        hi *= 2
        vhi = xnpv(hi)

    if not np.isfinite(vlo) or not np.isfinite(vhi) or vlo * vhi > 0:
        return np.nan

    for _ in range(400):
        mid = (lo + hi) / 2
        vm = xnpv(mid)
        if abs(vm) < 1e-12:
            return mid
        if vlo * vm <= 0:
            hi = mid
            vhi = vm
        else:
            lo = mid
            vlo = vm

    return (lo + hi) / 2


def compute_realized_dd(df_ledger: pd.DataFrame) -> float:
    if df_ledger is None or len(df_ledger) == 0:
        return np.nan
    s = df_ledger.sort_values(['Vente', 'Achat']).copy()
    cum = s['Gain'].cumsum()
    peak = cum.cummax()
    dd = peak - cum
    return float(dd.max())


def summarize_period(df_ledger: pd.DataFrame, start_year: int, end_year: int):
    if df_ledger is None or len(df_ledger) == 0:
        return {
            'trades': 0,
            'profit': 0.0,
            'xirr': np.nan,
            'dd': np.nan,
            'tp': 0,
            'be': 0,
            'sl': 0,
            'lock1': 0,
            'profit_factor': np.nan
        }

    vente_year = pd.to_datetime(df_ledger['Vente']).dt.year
    g = df_ledger[(vente_year >= start_year) & (vente_year <= end_year)].copy()

    if len(g) == 0:
        return {
            'trades': 0,
            'profit': 0.0,
            'xirr': np.nan,
            'dd': np.nan,
            'tp': 0,
            'be': 0,
            'sl': 0,
            'lock1': 0,
            'profit_factor': np.nan
        }

    wins = g.loc[g['Gain'] > 0, 'Gain']
    losses = g.loc[g['Gain'] < 0, 'Gain']

    return {
        'trades': int(len(g)),
        'profit': float(g['Gain'].sum()),
        'xirr': float(compute_xirr_from_ledger(g)),
        'dd': float(compute_realized_dd(g)),
        'tp': int((g['Type'] == 'TP').sum()),
        'be': int((g['Type'] == 'BE').sum()),
        'sl': int((g['Type'] == 'SL').sum()),
        'lock1': int((g['Type'] == 'LOCK1').sum()),
        'profit_factor': float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan
    }


def choose_best_variant(results_for_variants):
    """
    Règle simple:
    1. profit calibration max
    2. si égalité, DD calibration plus faible
    3. si égalité, XIRR calibration plus élevé
    """
    best = None
    for r in results_for_variants:
        if best is None:
            best = r
            continue

        a = r['train_metrics']
        b = best['train_metrics']

        if a['profit'] > b['profit']:
            best = r
        elif a['profit'] == b['profit']:
            if (np.isnan(b['dd']) or (not np.isnan(a['dd']) and a['dd'] < b['dd'])):
                best = r
            elif a['dd'] == b['dd']:
                if (np.isnan(b['xirr']) or (not np.isnan(a['xirr']) and a['xirr'] > b['xirr'])):
                    best = r
    return best


# =========================================================
# WALK-FORWARD AVEC CACHE
# =========================================================
def run_walkforward():
    all_fold_outputs = []

    # 1) Charger les données UNE SEULE FOIS
    base_df = load_market_data(ALPHA4_CFG)

    # 2) Calculer baseline + variantes UNE SEULE FOIS
    run_cache = {}

    # baseline
    base_cfg = copy.deepcopy(ALPHA4_CFG)
    base_cfg['USE_RANGE_SLOW_CONTEXT_FILTER'] = False
    base_run = alpha4(base_cfg, base_df=base_df)
    run_cache['BASELINE'] = {
        'config': {
            'use_filter': False,
            'score_mode': None,
            'max_idx_gap': None
        },
        'metadata': base_run['metadata'],
        'ledger': pd.DataFrame(base_run['trades'])
    }

    # variantes
    for var in VARIANTS:
        cfg = copy.deepcopy(ALPHA4_CFG)
        cfg['USE_RANGE_SLOW_CONTEXT_FILTER'] = True
        cfg['RANGE_SLOW_SCORE_MODE'] = var['RANGE_SLOW_SCORE_MODE']
        cfg['RANGE_SLOW_MAX_IDX_GAP'] = var['RANGE_SLOW_MAX_IDX_GAP']

        run = alpha4(cfg, base_df=base_df)

        run_cache[var['name']] = {
            'config': {
                'use_filter': True,
                'score_mode': var['RANGE_SLOW_SCORE_MODE'],
                'max_idx_gap': var['RANGE_SLOW_MAX_IDX_GAP']
            },
            'metadata': run['metadata'],
            'ledger': pd.DataFrame(run['trades'])
        }

    # 3) Walk-forward à partir des ledgers déjà calculés
    for fold in WALKFORWARD_FOLDS:
        fold_name = fold['name']
        train_start = fold['train_start']
        train_end = fold['train_end']
        test_year = fold['test_year']

        baseline_ledger = run_cache['BASELINE']['ledger']
        baseline_train = summarize_period(baseline_ledger, train_start, train_end)
        baseline_test = summarize_period(baseline_ledger, test_year, test_year)

        variant_results = []

        for var in VARIANTS:
            variant_name = var['name']
            ledger = run_cache[variant_name]['ledger']

            train_metrics = summarize_period(ledger, train_start, train_end)
            test_metrics = summarize_period(ledger, test_year, test_year)

            variant_results.append({
                'variant_name': variant_name,
                'config': run_cache[variant_name]['config'],
                'metadata': run_cache[variant_name]['metadata'],
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            })

        best_variant = choose_best_variant(variant_results)

        fold_output = {
            'fold': fold_name,
            'train_period': f'{train_start}-{train_end}',
            'test_period': str(test_year),
            'baseline': {
                'train': baseline_train,
                'test': baseline_test
            },
            'all_variants': variant_results,
            'selected_variant': best_variant
        }

        all_fold_outputs.append(fold_output)

    return all_fold_outputs


# =========================================================
# SORTIE RESUMEE POUR EVITER UN JSON TROP GROS
# =========================================================
def build_compact_summary(wf_results):
    summary = []
    for fold in wf_results:
        summary.append({
            'fold': fold['fold'],
            'train_period': fold['train_period'],
            'test_period': fold['test_period'],
            'baseline_test': fold['baseline']['test'],
            'selected_variant': {
                'variant_name': fold['selected_variant']['variant_name'],
                'config': fold['selected_variant']['config'],
                'metadata': {
                    'total_skipped_range_slow_context': fold['selected_variant']['metadata'].get('total_skipped_range_slow_context'),
                    'total_skipped_tp135_slow': fold['selected_variant']['metadata'].get('total_skipped_tp135_slow')
                },
                'train_metrics': fold['selected_variant']['train_metrics'],
                'test_metrics': fold['selected_variant']['test_metrics']
            }
        })
    return summary


if __name__ == '__main__':
    wf = run_walkforward()

    # Résumé compact recommandé
    compact = build_compact_summary(wf)
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=str))

    # Si tu veux le JSON complet, décommente :
    # print(json.dumps(wf, indent=2, ensure_ascii=False, default=str))

# ==== END sigma2.py ====
