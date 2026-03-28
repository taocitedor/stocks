# === v10 - 28032026
# ==== START sigma2.py
import math
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery

ALPHA4_CFG = {
    # --- Configuration Backend ---
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours_v2',
    'IDX': '^FCHI',

    # --- Fenêtre d'analyse simple ---
    'USE_DAYS_BACK_FILTER': False,   # False = tout l'historique / True = filtre actif
    'DAYS_BACK_FROM_TODAY': 365,     # nombre de jours en arrière depuis aujourd'hu

    # --- Gestion portefeuille / cash ---
    'INITIAL_CASH': 50000.0,             # cash de départ
    'USE_CASH_ALLOCATOR': True,          # active la couche portefeuille
    
    # Taille des nouvelles positions
    'POSITION_SIZE_MODE': 'fixed',       # 'fixed' ou 'equal_split'
    'SIZE': 4000.0,                      # cible nominale par ligne
    'MIN_ORDER_EUR': 1000.0,             # pas de ligne si trop petite
    
    # Contraintes portefeuille
    'MAX_OPEN_POSITIONS': 10,
    'MAX_TOTAL_EXPOSURE_PCT': 0.80,      # 80% du capital max investi
    'MIN_CASH_BUFFER_PCT': 0.05,         # garde 5% de cash
    'MAX_NEW_ENTRIES_PER_DAY': 3,        # max de nouvelles entrées par jour
    
    # Priorisation des signaux concurrents le même jour
    'ENTRY_PRIORITY': 'score_then_volume',   # 'score_only' ou 'score_then_volume'

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

    # --- Filtre régime ---
    'EXCLUDE_TP135_SLOW': True,

    # --- Univers ---
    'UNIVERSE': None
}

# ===========================
# Indicateurs (parité GAS)
# ===========================
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
    
# ===========================
# Fonctions utilitaires
# ===========================

def _to_ts(x):
    ts = pd.to_datetime(x, errors='coerce')
    if pd.isna(ts):
        return pd.NaT
    try:
        return ts.tz_localize(None)
    except Exception:
        return ts

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _sort_trade_candidates(candidates, cfg):
    mode = cfg.get('ENTRY_PRIORITY', 'score_then_volume')

    if mode == 'score_only':
        return sorted(
            candidates,
            key=lambda x: _safe_float(x.get('Score_Entry'), -1e9),
            reverse=True
        )

    return sorted(
        candidates,
        key=lambda x: (
            _safe_float(x.get('Score_Entry'), -1e9),
            _safe_float(x.get('Volume_Ratio_Entry'), -1e9)
        ),
        reverse=True
    )

def _compute_real_entry_quantity(entry_px, budget_eur, fees_pct):
    """
    Calcule la quantité entière maximale d'actions achetables
    sans dépasser le budget, frais inclus.
    """
    entry_px = _safe_float(entry_px, 0.0)
    budget_eur = _safe_float(budget_eur, 0.0)
    fees_pct = _safe_float(fees_pct, 0.0)

    if entry_px <= 0 or budget_eur <= 0:
        return 0, 0.0, 0.0, 0.0

    cost_per_share = entry_px * (1.0 + fees_pct)
    qty = int(math.floor(budget_eur / cost_per_share))

    if qty <= 0:
        return 0, 0.0, 0.0, 0.0

    gross_buy = qty * entry_px
    buy_fees = gross_buy * fees_pct
    cash_debited = gross_buy + buy_fees

    return qty, gross_buy, buy_fees, cash_debited

def _compute_real_exit_cash(qty, exit_px, fees_pct):
    """
    Cash récupéré à la vente après frais.
    """
    qty = _safe_int(qty, 0)
    exit_px = _safe_float(exit_px, 0.0)
    fees_pct = _safe_float(fees_pct, 0.0)

    if qty <= 0 or exit_px <= 0:
        return 0.0, 0.0, 0.0

    gross_sell = qty * exit_px
    sell_fees = gross_sell * fees_pct
    cash_credited = gross_sell - sell_fees

    return gross_sell, sell_fees, cash_credited

# ===========================
# Moteur par ticker
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

    # --- RS momentum ---
    rs_sma_p = int(cfg.get('RS_SMA_P', 20))
    rs_sma = rs_line.rolling(window=rs_sma_p).mean()

    if cfg.get('USE_RS_SMA_FILTER', False):
        rs_momentum_ok = (rs_line > rs_sma)
    else:
        rs_momentum_ok = pd.Series(True, index=stock_df.index)

    # --- Score ---
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

    ledger = []
    active_trade = None

    start_i = max(
        cfg['SMA_P'],
        cfg.get('PRICE_SMA_P', 200) if cfg.get('USE_PRICE_SMA_FILTER', False) else 0
    )

    skipped_tp135_slow = 0

    # --- Boucle simulation ---
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

            # --- Profit Lock ---
            if cfg.get('USE_PROFIT_LOCK', False):
                if active_trade['mfe_pct'] >= cfg.get('LOCK2_TRIGGER', 0.10):
                    active_trade['profit_lock_raw'] = cfg.get('LOCK2_RAW', 0.0356)
                    active_trade['profit_lock_level'] = 'LOCK2'
                elif active_trade['mfe_pct'] >= cfg.get('LOCK1_TRIGGER', 0.095):
                    if active_trade['profit_lock_raw'] is None:
                        active_trade['profit_lock_raw'] = cfg.get('LOCK1_RAW', 0.0206)
                        active_trade['profit_lock_level'] = 'LOCK1'

            # Stop effectif
            effective_sl = cfg['FEES'] if active_trade['be_hit'] else -cfg['STOP_L']

            # Si profit lock actif, il remonte le stop si plus protecteur
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

                # --- benchmark à la sortie ---
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

                    # --- Logging analytique existant ---
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

                    # --- Profit Lock existant ---
                    'Profit_Lock_Level': active_trade.get('profit_lock_level', None),
                    'Profit_Lock_Raw_Pct': round(active_trade['profit_lock_raw'] * 100, 2)
                    if active_trade.get('profit_lock_raw') is not None else None,

                    # ======================================================
                    # NOUVEAUX CHAMPS D'ANALYSE A L'ENTREE
                    # ======================================================
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

                    # ======================================================
                    # PERFORMANCE RELATIVE AU BENCHMARK
                    # ======================================================
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

            # ======================================================
            # Filtre demandé : exclure TP13.5 + SLOW
            # ======================================================
            if cfg.get('EXCLUDE_TP135_SLOW', False):
                is_tp_135 = abs(current_tp - cfg['TP_TREND']) < 1e-12
                is_slow = not is_fast_be
                if is_tp_135 and is_slow:
                    skipped_tp135_slow += 1
                    continue

            # --- enrichissement contexte d'entrée ---
            tp_regime_source = 'TREND'
            if is_strong:
                tp_regime_source = 'TREND_BOOST'
            if slope < cfg['SLOPE_TRESH']:
                tp_regime_source = 'RANGE'

            idx_entry_px = idx_close.reindex(stock_df.index).loc[date]
            idx_entry_sma = idx_sma_on_stock_dates.reindex(stock_df.index).loc[date]

            idx_gap_vs_sma_pct = None
            if pd.notna(idx_entry_px) and pd.notna(idx_entry_sma) and idx_entry_sma != 0:
                idx_gap_vs_sma_pct = ((float(idx_entry_px) / float(idx_entry_sma)) - 1.0) * 100.0

            price_sma_entry = price_sma.loc[date] if date in price_sma.index else np.nan
            price_vs_sma200_pct = None
            if pd.notna(price_sma_entry) and price_sma_entry != 0:
                price_vs_sma200_pct = ((float(row['Close']) / float(price_sma_entry)) - 1.0) * 100.0

            rs_sma_entry = rs_sma.loc[date] if date in rs_sma.index else np.nan

            active_trade = {
                'date': date,
                'e_px': float(row['Close']),           
                'size': float(cfg['SIZE']),
                'fees': float(cfg['FEES']),
                'tp_val': float(current_tp),
                'be_trig': float(current_be_trig),
                'be_type': 'FAST' if is_fast_be else 'SLOW',
                'be_hit': False,
                'bars_held': 0,

                # --- Profit lock state ---
                'profit_lock_raw': None,
                'profit_lock_level': None,

                # --- Logging analytique existant ---
                'mfe_pct': 0.0,
                'mae_pct': 0.0,
                'max_close_pct': 0.0,
                'min_close_pct': 0.0,
                'bars_to_be': None,
                'bars_to_tp': None,
                'bars_to_sl': None,

                # ======================================================
                # NOUVEAUX CHAMPS D'ANALYSE A L'ENTREE
                # ======================================================
                'Score_Entry': float(score),
                'rs_line_entry': float(rs_line.loc[date]),
                'rs_sma_entry': float(rs_sma_entry) if pd.notna(rs_sma_entry) else None,
                'rs_momentum_ok_entry': bool(rs_momentum_ok.loc[date]),
                'rsi_entry': float(rsi.loc[date]),
                'Volume_Ratio_Entry': float(row.get('Volume_Ratio', 0.0)) if 'Volume_Ratio' in row else None,
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
                'Structure_Label_Entry': structure_label,
                'tp_regime_source': tp_regime_source
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
            'Bars_to_BE': active_trade['bars_to_be'],

            # --- Profit lock ---
            'Profit_Lock_Level': active_trade.get('profit_lock_level', None),
            'Profit_Lock_Raw_Pct': round(active_trade['profit_lock_raw'] * 100, 2)
            if active_trade.get('profit_lock_raw') is not None else None
        }

    df_ledger = pd.DataFrame(ledger)

    stats = {
        'nb_trades': int(len(df_ledger)),
        'gain_total': float(df_ledger['Gain'].sum()) if len(df_ledger) else 0.0,
        'win_rate': float((df_ledger['Gain'] > 0).mean()) if len(df_ledger) else 0.0,
        'skipped_tp135_slow': int(skipped_tp135_slow)
    }

    return stats, ledger, open_trade

# ===========================
# Cloture - gestion de portefeuille
# ===========================

def _close_trade_v4(tr, date, exit_px, exit_type, stock_df_attrs, idx_close_entry=None, idx_close_exit=None):
    """
    Clôture un trade en conservant explicitement prix d'entrée / sortie
    et toutes les infos utiles à l'allocator portefeuille.
    """
    entry_px = _safe_float(tr.get('e_px'), 0.0)
    exit_px = _safe_float(exit_px, 0.0)
    fees = _safe_float(tr.get('fees', 0.0), 0.0)
    size = _safe_float(tr.get('size', 0.0), 0.0)

    # PnL théorique historique sur SIZE nominal
    gross_buy = size
    cash_out = gross_buy * (1.0 + fees)

    qty_theoretical = gross_buy / entry_px if entry_px > 0 else 0.0
    gross_sell = qty_theoretical * exit_px if exit_px > 0 else 0.0
    cash_in = gross_sell * (1.0 - fees)

    gain = cash_in - cash_out

    stock_ret_trade_pct = ((exit_px / entry_px) - 1.0) * 100.0 if entry_px > 0 and exit_px > 0 else None
    idx_return_trade_pct = None
    excess_return_vs_idx_pct = None

    if idx_close_entry is not None and idx_close_exit is not None and idx_close_entry > 0:
        idx_return_trade_pct = ((idx_close_exit / idx_close_entry) - 1.0) * 100.0
        if stock_ret_trade_pct is not None:
            excess_return_vs_idx_pct = stock_ret_trade_pct - idx_return_trade_pct

    out = {
        'Ticker': stock_df_attrs.get('Ticker', 'NA'),
        'Achat': tr['date'].strftime('%Y-%m-%d'),
        'Vente': date.strftime('%Y-%m-%d'),

        'Prix_Entree': round(entry_px, 6),
        'Prix_Vente': round(exit_px, 6),

        'Type': exit_type,
        'Gain': round(gain, 2),
        'Orig_SIZE': round(size, 2),

        'Stock_Return_Trade_Pct': round(stock_ret_trade_pct, 4) if stock_ret_trade_pct is not None else None,
        'Idx_Return_Trade_Pct': round(idx_return_trade_pct, 4) if idx_return_trade_pct is not None else None,
        'Excess_Return_vs_Idx_Pct': round(excess_return_vs_idx_pct, 4) if excess_return_vs_idx_pct is not None else None,

        'BE_Assigned_Pct': round(_safe_float(tr.get('be_trig'), 0.0) * 100.0, 4),
        'BE_Type': tr.get('be_type'),
        'TP_Assigned_Pct': round(_safe_float(tr.get('tp_val'), 0.0) * 100.0, 4),

        'Bars': tr.get('bars_held'),
        'Bars_to_BE': tr.get('bars_to_be'),
        'Bars_to_SL': tr.get('bars_to_sl'),
        'Bars_to_TP': tr.get('bars_to_tp'),

        'MAE_Pct': round(_safe_float(tr.get('mae_pct'), 0.0) * 100.0, 4),
        'MFE_Pct': round(_safe_float(tr.get('mfe_pct'), 0.0) * 100.0, 4),
        'Max_Close_Pct': round(_safe_float(tr.get('max_close_pct'), 0.0) * 100.0, 4),
        'Min_Close_Pct': round(_safe_float(tr.get('min_close_pct'), 0.0) * 100.0, 4),

        'Score_Entry': tr.get('Score_Entry'),
        'Volume_Ratio_Entry': tr.get('Volume_Ratio_Entry'),
        'TP_Regime_Source': tr.get('TP_Regime_Source'),
        'Structure_Label_Entry': tr.get('Structure_Label_Entry'),

        'Profit_Lock_Level': tr.get('profit_lock_level'),
        'Profit_Lock_Raw_Pct': round(_safe_float(tr.get('profit_lock_raw'), 0.0) * 100.0, 4)
        if tr.get('profit_lock_raw') is not None else None
    }

    return out

# ===========================
# Gestion du cash
# ===========================

def _apply_cash_allocator(candidate_trades, candidate_open_positions, cfg):
    if not candidate_trades:
        return [], [], {
            'initial_cash': _safe_float(cfg.get('INITIAL_CASH', 0.0)),
            'ending_cash': _safe_float(cfg.get('INITIAL_CASH', 0.0)),
            'max_exposure_eur': 0.0,
            'max_open_positions_realized': 0,
            'rejected_entries_count': 0,
            'exposure_cap_eur': 0.0,
            'cash_buffer_eur': 0.0
        }

    initial_cash = _safe_float(cfg.get('INITIAL_CASH', 50000.0))
    cash = initial_cash

    size_cfg = _safe_float(cfg.get('SIZE', 4000.0))
    fees_pct = _safe_float(cfg.get('FEES', 0.0))
    min_order_eur = _safe_float(cfg.get('MIN_ORDER_EUR', 1000.0))
    max_open_positions = int(cfg.get('MAX_OPEN_POSITIONS', 10))
    max_new_entries_per_day = int(cfg.get('MAX_NEW_ENTRIES_PER_DAY', 3))
    max_total_exposure_pct = _safe_float(cfg.get('MAX_TOTAL_EXPOSURE_PCT', 0.80))
    min_cash_buffer_pct = _safe_float(cfg.get('MIN_CASH_BUFFER_PCT', 0.05))
    position_size_mode = cfg.get('POSITION_SIZE_MODE', 'fixed')

    cash_buffer = initial_cash * min_cash_buffer_pct
    exposure_cap_eur = initial_cash * max_total_exposure_pct

    trades = []
    for t in candidate_trades:
        tt = dict(t)
        tt['Achat_ts'] = _to_ts(tt.get('Achat'))
        tt['Vente_ts'] = _to_ts(tt.get('Vente'))
        if pd.isna(tt['Achat_ts']) or pd.isna(tt['Vente_ts']):
            continue

        tt['Entry_Price_Real'] = _safe_float(tt.get('Prix_Entree'), 0.0)
        tt['Exit_Price_Real'] = _safe_float(tt.get('Prix_Vente'), 0.0)
        trades.append(tt)

    if not trades:
        return [], [], {
            'initial_cash': initial_cash,
            'ending_cash': initial_cash,
            'max_exposure_eur': 0.0,
            'max_open_positions_realized': 0,
            'rejected_entries_count': 0,
            'exposure_cap_eur': exposure_cap_eur,
            'cash_buffer_eur': cash_buffer
        }

    all_dates = sorted(set(
        [t['Achat_ts'].normalize() for t in trades] +
        [t['Vente_ts'].normalize() for t in trades]
    ))

    entries_by_date = defaultdict(list)
    exits_by_date = defaultdict(list)

    for t in trades:
        entries_by_date[t['Achat_ts'].normalize()].append(t)
        exits_by_date[t['Vente_ts'].normalize()].append(t)

    open_positions = []
    accepted_trades = []
    rejected_entries_count = 0

    max_exposure_realized = 0.0
    max_open_positions_realized = 0

    for current_date in all_dates:
        # =================================================
        # 1) Sorties : on libère le cash d'abord
        # =================================================
        still_open = []
        for pos in open_positions:
            if pos['Vente_ts'].normalize() == current_date:
                gross_sell, sell_fees, cash_credited = _compute_real_exit_cash(
                    qty=pos['Qty'],
                    exit_px=pos['Exit_Price_Real'],
                    fees_pct=fees_pct
                )

                cash += cash_credited

                pos['Gross_Sell_EUR'] = round(gross_sell, 2)
                pos['Sell_Fees_EUR'] = round(sell_fees, 2)
                pos['Cash_Credited_At_Exit_EUR'] = round(cash_credited, 2)
                pos['Allocated_Gain_EUR'] = round(cash_credited - pos['Cash_Debited_At_Entry_EUR'], 2)

                accepted_trades.append(pos)
            else:
                still_open.append(pos)

        open_positions = still_open

        # =================================================
        # 2) Budget disponible
        # =================================================
        current_exposure = sum(p['Gross_Buy_EUR'] for p in open_positions)
        remaining_exposure_cap = max(0.0, exposure_cap_eur - current_exposure)
        available_cash_for_entries = max(0.0, cash - cash_buffer)
        entry_budget = min(available_cash_for_entries, remaining_exposure_cap)
        remaining_slots = max_open_positions - len(open_positions)

        max_exposure_realized = max(max_exposure_realized, current_exposure)
        max_open_positions_realized = max(max_open_positions_realized, len(open_positions))

        if entry_budget <= 0 or remaining_slots <= 0:
            continue

        # =================================================
        # 3) Signaux du jour
        # =================================================
        day_candidates = _sort_trade_candidates(entries_by_date.get(current_date, []), cfg)
        if not day_candidates:
            continue

        day_candidates = day_candidates[:max_new_entries_per_day]
        day_candidates = day_candidates[:remaining_slots]

        if not day_candidates:
            continue

        # =================================================
        # 4) Allocation de budget entre signaux retenus
        # =================================================
        if position_size_mode == 'equal_split':
            target_budgets = [entry_budget / len(day_candidates)] * len(day_candidates)
        else:
            target_budgets = []
            tmp_budget = entry_budget
            for _ in day_candidates:
                budget_i = min(size_cfg, tmp_budget)
                target_budgets.append(budget_i)
                tmp_budget -= budget_i

        # =================================================
        # 5) Entrées réelles : quantité entière d'actions
        # =================================================
        for cand, target_budget in zip(day_candidates, target_budgets):
            entry_px = _safe_float(cand.get('Entry_Price_Real'), 0.0)
            if entry_px <= 0:
                rejected_entries_count += 1
                continue

            qty, gross_buy, buy_fees, cash_debited = _compute_real_entry_quantity(
                entry_px=entry_px,
                budget_eur=min(target_budget, entry_budget),
                fees_pct=fees_pct
            )

            if qty <= 0 or cash_debited < min_order_eur:
                rejected_entries_count += 1
                continue

            if cash_debited > cash:
                rejected_entries_count += 1
                continue

            cash -= cash_debited
            entry_budget -= cash_debited

            pos = dict(cand)
            pos['Qty'] = int(qty)
            pos['Gross_Buy_EUR'] = round(gross_buy, 2)
            pos['Buy_Fees_EUR'] = round(buy_fees, 2)
            pos['Cash_Debited_At_Entry_EUR'] = round(cash_debited, 2)
            pos['Allocated_SIZE_EUR'] = round(gross_buy, 2)
            pos['Portfolio_Entry_Date'] = current_date.strftime('%Y-%m-%d')
            pos['Portfolio_Cash_After_Entry'] = round(cash, 2)

            open_positions.append(pos)

        current_exposure = sum(p['Gross_Buy_EUR'] for p in open_positions)
        max_exposure_realized = max(max_exposure_realized, current_exposure)
        max_open_positions_realized = max(max_open_positions_realized, len(open_positions))

    # =====================================================
    # 6) Positions encore ouvertes en fin de période
    # =====================================================
    accepted_open_positions = []

    candidate_open_map = {}
    for op in candidate_open_positions:
        key = (
            str(op.get('Ticker')),
            str(op.get('Date_Achat'))
        )
        candidate_open_map[key] = op

    for pos in open_positions:
        key = (str(pos.get('Ticker')), str(pos.get('Achat'))[:10])
        enriched = candidate_open_map.get(key)

        if enriched:
            final_op = dict(enriched)
        else:
            final_op = {
                'Ticker': pos.get('Ticker'),
                'Date_Achat': str(pos.get('Achat'))[:10],
                'Prix_Entree': pos.get('Entry_Price_Real'),
                'Prix_Actuel': None,
                'Perf_Latente_Pct': None
            }

        final_op['Qty'] = pos['Qty']
        final_op['Allocated_SIZE_EUR'] = pos['Allocated_SIZE_EUR']
        final_op['Gross_Buy_EUR'] = pos['Gross_Buy_EUR']
        final_op['Buy_Fees_EUR'] = pos['Buy_Fees_EUR']
        final_op['Cash_Debited_At_Entry_EUR'] = pos['Cash_Debited_At_Entry_EUR']
        final_op['Portfolio_Cash_Current'] = round(cash, 2)

        accepted_open_positions.append(final_op)

    allocator_metadata = {
        'initial_cash': round(initial_cash, 2),
        'ending_cash': round(cash, 2),
        'max_exposure_eur': round(max_exposure_realized, 2),
        'max_open_positions_realized': int(max_open_positions_realized),
        'rejected_entries_count': int(rejected_entries_count),
        'exposure_cap_eur': round(exposure_cap_eur, 2),
        'cash_buffer_eur': round(cash_buffer, 2)
    }

    return accepted_trades, accepted_open_positions, allocator_metadata


# ===========================
# Moteur multi-tickers
# ===========================
def alpha4(cfg):
    client = bigquery.Client(project=cfg['PROJECT'])

    if cfg.get('USE_DAYS_BACK_FILTER', False):
        days_back = int(cfg.get('DAYS_BACK_FROM_TODAY', 365))
        days_back = max(days_back, 365)
        start_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')

        query = f"""
            SELECT *
            FROM `{cfg['DB_SET']}.{cfg['TBL']}`
            WHERE Date >= DATE('{start_date}')
            ORDER BY Date ASC
        """
    else:
        query = f"SELECT * FROM `{cfg['DB_SET']}.{cfg['TBL']}` ORDER BY Date ASC"

    df = client.query(query).to_dataframe()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    for c in ['Close', 'High', 'Low', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Déduplication sécurité
    df = df.sort_values(['Ticker', 'Date'])
    df = df.drop_duplicates(subset=['Ticker', 'Date'], keep='last').reset_index(drop=True)

    idx_ticker = cfg['IDX']
    base_idx = (
        df[df['Ticker'] == idx_ticker]
        .copy()
        .drop_duplicates(subset=['Date'], keep='last')
        .set_index('Date')
        .sort_index()
    )

    idx_close = base_idx['Close']
    idx_sma = idx_close.rolling(cfg['SMA_P'], min_periods=cfg['SMA_P']).mean()
    idx_slope = ((idx_sma - idx_sma.shift(4)) / idx_sma.shift(4)).fillna(0)

    universe = [t for t in sorted(df['Ticker'].dropna().unique()) if t != idx_ticker]
    if cfg['UNIVERSE']:
        universe = [t for t in universe if t in cfg['UNIVERSE']]

    all_candidate_trades = []
    all_candidate_open_positions = []
    per_ticker_stats = {}

    for t in universe:
        d = (
            df[df['Ticker'] == t]
            .copy()
            .drop_duplicates(subset=['Ticker', 'Date'], keep='last')
            .set_index('Date')
            .sort_index()
        )
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
        all_candidate_trades.extend(trades)

        if open_trade:
            all_candidate_open_positions.append(open_trade)

    if cfg.get('USE_CASH_ALLOCATOR', True):
        portfolio_trades, portfolio_open_positions, allocator_metadata = _apply_cash_allocator(
            all_candidate_trades,
            all_candidate_open_positions,
            cfg
        )
    else:
        portfolio_trades = all_candidate_trades
        portfolio_open_positions = all_candidate_open_positions
        allocator_metadata = {
            'initial_cash': None,
            'ending_cash': None,
            'max_exposure_eur': None,
            'max_open_positions_realized': None,
            'rejected_entries_count': 0,
            'exposure_cap_eur': None,
            'cash_buffer_eur': None
        }

    df_ledger = pd.DataFrame(portfolio_trades)
    total_skipped = int(sum(v.get('skipped_tp135_slow', 0) for v in per_ticker_stats.values()))

    gain_col = 'Allocated_Gain_EUR' if 'Allocated_Gain_EUR' in df_ledger.columns else 'Gain'

    return {
        'metadata': {
            'system': 'Titanium v7 + cash allocator + real share allocation',
            'universe': len(universe),
            'exclude_tp135_slow': cfg.get('EXCLUDE_TP135_SLOW', False),
            'total_skipped_tp135_slow': total_skipped,
            'use_days_back_filter': cfg.get('USE_DAYS_BACK_FILTER', False),
            'days_back_from_today': cfg.get('DAYS_BACK_FROM_TODAY', None) if cfg.get('USE_DAYS_BACK_FILTER', False) else None,
            'use_cash_allocator': cfg.get('USE_CASH_ALLOCATOR', True)
        },
        'portfolio': {
            'gain_total': float(df_ledger[gain_col].sum()) if len(df_ledger) else 0.0,
            'nb_trades': int(len(df_ledger)) if len(df_ledger) else 0,
            'win_rate': float((df_ledger[gain_col] > 0).mean()) if len(df_ledger) else 0.0,
            'by_ticker': per_ticker_stats,
            'allocator': allocator_metadata
        },
        'open_positions': portfolio_open_positions,
        'trades': df_ledger.to_dict(orient='records') if len(df_ledger) else []
    }


if __name__ == '__main__':
    out = alpha4(ALPHA4_CFG)
    print(json.dumps(out, indent=2, ensure_ascii=False))

# ==== END sigma2.py
