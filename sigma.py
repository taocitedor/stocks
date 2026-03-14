import pandas as pd
import numpy as np
from google.cloud import bigquery
import json

# --- CONFIGURATION SYSTÈME ALPHA ---
ALPHA_CFG = {
    'PROJECT': 'project-16c606d0-6527-4644-907',
    'DB_SET': 'Trading',
    'TBL': 'CC_Historique_Cours',
    'STOCK': 'EN.PA',
    'IDX': '^FCHI',
    'MKT_FILTER': True,
    'SMA_P': 100,
    'MIN_SCORE': 86,
    'LOOKBACK': 63,   # comme GAS: fenêtre de 63 points => comparaison current vs current-62
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
    'DEBUG_DATE': '2025-07-29'
}


# =============================================================================
# RS 100% aligné sur la logique GAS
# =============================================================================
def gas_rs_series(stock_close: pd.Series, idx_close: pd.Series) -> pd.Series:
    """
    Réplique de la logique GAS:

      if (!idx || s.length < 63) return 0;
      let p0s = s[s.length-1].close, currentTs = s[s.length-1].ts;
      let idxD = idx.filter(i => i.ts <= currentTs).sort((a,b) => b.ts - a.ts);
      return (idxD.length < 63) ? 0 :
        ((p0s - s[s.length-63].close) / s[s.length-63].close
       - (idxD[0].close - idxD[62].close) / idxD[62].close) * 100;

    Important:
    - côté stock, avec length=63, GAS compare current vs current-62
    - côté indice, prend le dernier close dispo <= date du stock, puis remonte 62 barres
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

        # dernier point d'indice disponible <= current_dt
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

    return pd.Series(out, index=stock_close.index, name="RS_Line")


# =============================================================================
# RSI aligné sur GAS
# =============================================================================
def gas_rsi_series(close: pd.Series, p: int = 14) -> pd.Series:
    """
    Réplique l'esprit de:
      let g = 0, pt = 0;
      for (let i = cls.length - p; i < cls.length; i++) {
        let d = cls[i] - cls[i-1];
        if (d > 0) g += d; else pt -= d;
      }
      return pt === 0 ? 100 : 100 - (100 / (1 + (g / pt)));

    rolling(p).sum() sur les diffs correspond à la logique GAS.
    """
    close = pd.to_numeric(close, errors='coerce')
    diff = close.diff()

    gains = diff.clip(lower=0).rolling(p, min_periods=p).sum()
    losses = (-diff.clip(upper=0)).rolling(p, min_periods=p).sum()

    rsi = pd.Series(np.nan, index=close.index, dtype=float)

    # cas GAS: pertes == 0 => 100
    zero_losses = losses == 0
    normal = ~zero_losses & losses.notna()

    rsi[zero_losses] = 100.0
    rsi[normal] = 100 - (100 / (1 + (gains[normal] / losses[normal])))

    # avant disponibilité complète => NaN, puis on met 0 comme ton pipeline de score
    return rsi.fillna(0.0)


# =============================================================================
# Pivots exact type GAS (événements chronologiques)
# =============================================================================
def gas_pivots_events(df: pd.DataFrame, w: int = 3):
    """
    Réplique exacte de:
      function VLAB_UTIL_Pivots(ser, w) {
        let p = [];
        for (let i = w; i < ser.length - w; i++) {
          let isH = true, isL = true;
          for (let j = i - w; j <= i + w; j++) {
            if (j == i) continue;
            if (ser[j].high >= ser[i].high) isH = false;
            if (ser[j].low <= ser[i].low) isL = false;
          }
          if (isH) p.push({type: "H", value: ser[i].high});
          if (isL) p.push({type: "L", value: ser[i].low});
        }
        return p;
      }
    """
    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    pivots = []

    for i in range(w, n - w):
        isH = True
        isL = True

        for j in range(i - w, i + w + 1):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                isH = False
            if lows[j] <= lows[i]:
                isL = False
            if not isH and not isL:
                break

        if isH:
            pivots.append({
                "pivot_i": i,
                "type": "H",
                "value": float(highs[i])
            })
        if isL:
            pivots.append({
                "pivot_i": i,
                "type": "L",
                "value": float(lows[i])
            })

    return pivots


# =============================================================================
# Structure 100% alignée sur GAS
# =============================================================================
def gas_structure_series(df: pd.DataFrame, w: int = 3, last_pivots: int = 15):
    """
    Réplique:
      let struct = VLAB_UTIL_Structure(VLAB_UTIL_Pivots(subS, 3).slice(-15));

      function VLAB_UTIL_Structure(p) {
        let h = p.filter(x => x.type == "H"), l = p.filter(x => x.type == "L");
        if (h.length < 2 || l.length < 2) return "ND";
        return (h[h.length-1].value > h[h.length-2].value ? "HH" : "LH")
             + "+"
             + (l[l.length-1].value > l[l.length-2].value ? "HL" : "LL");
      }

    Point clé:
    - un pivot détecté à l’index k n’est visible dans subS qu’à partir de i = k + w
      (car le pivot a besoin de w barres à droite pour être confirmé)
    - à chaque date i, on prend les pivots visibles à date, puis slice(-15)
    """
    df = df.sort_index().copy()
    n = len(df)

    all_pivots = gas_pivots_events(df, w=w)

    # visible_on[i] = pivots qui deviennent visibles à la date i
    visible_on = [[] for _ in range(n)]
    for p in all_pivots:
        vis_i = p["pivot_i"] + w
        if vis_i < n:
            visible_on[vis_i].append(p)

    active_pivots = []
    struct_label = np.array(["ND"] * n, dtype=object)
    struct_ok = np.zeros(n, dtype=bool)

    for i in range(n):
        if visible_on[i]:
            active_pivots.extend(visible_on[i])

        p15 = active_pivots[-last_pivots:]
        h = [x for x in p15 if x["type"] == "H"]
        l = [x for x in p15 if x["type"] == "L"]

        if len(h) < 2 or len(l) < 2:
            struct_label[i] = "ND"
            struct_ok[i] = False
            continue

        struct = (
            ("HH" if h[-1]["value"] > h[-2]["value"] else "LH")
            + "+"
            + ("HL" if l[-1]["value"] > l[-2]["value"] else "LL")
        )

        struct_label[i] = struct
        struct_ok[i] = ("HH" in struct and "HL" in struct)

    return (
        pd.Series(struct_label, index=df.index, name="Structure"),
        pd.Series(struct_ok, index=df.index, name="Structure_OK")
    )


# =============================================================================
# Squeeze aligné sur GAS
# =============================================================================
def gas_squeeze_series(df: pd.DataFrame) -> pd.Series:
    """
    GAS:
      let l20 = ser.slice(-20), cls = l20.map(s => s.close);
      let sma = avg20(close)
      let sd = std20(close) with divisor 20 (population std)
      let atr = avg20(high-low)
      return (sma + 2*sd < sma + 1.5*atr) && (sma - 2*sd > sma - 1.5*atr)

    => équivalent simplifié:
       2*sd < 1.5*atr
    """
    cls = pd.to_numeric(df["Close"], errors="coerce")
    hl = pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")

    std20 = cls.rolling(20, min_periods=20).std(ddof=0)  # population std
    atr20_hl = hl.rolling(20, min_periods=20).mean()

    return ((2 * std20) < (1.5 * atr20_hl)).fillna(False)


# =============================================================================
# ATR "classique" conservé dans ton moteur v3
# =============================================================================
def atr_true_range_series(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df['High'], errors='coerce')
    low = pd.to_numeric(df['Low'], errors='coerce')
    close_prev = pd.to_numeric(df['Close'], errors='coerce').shift(1)

    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)

    return tr


# =============================================================================
# MOTEUR PRINCIPAL
# =============================================================================
def alpha_engine_v3():
    # 1) ACQUISITION & NETTOYAGE
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
    base_idx = raw_df[raw_df['Ticker'] == ALPHA_CFG['IDX']].set_index('Date').copy()

    for df in [base_stock, base_idx]:
        for c in ['Close', 'High', 'Low', 'Volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    base_stock = base_stock.sort_index()
    base_idx = base_idx.sort_index()

    # 2) INDICATEURS
    # RS 100% alignée GAS
    rs_line = gas_rs_series(
        stock_close=base_stock['Close'],
        idx_close=base_idx['Close']
    )

    # RSI aligné GAS
    rsi_gold = gas_rsi_series(base_stock['Close'], p=14)

    # Volume ratio
    v_ratio = (base_stock['Volume'] / base_stock['Volume'].rolling(20, min_periods=20).mean()).fillna(0)

    # MM20 + distance
    mm20 = base_stock['Close'].rolling(20, min_periods=20).mean()
    dist_mm20 = ((base_stock['Close'] - mm20).abs() / mm20).fillna(1)

    # Squeeze
    is_sqz = gas_squeeze_series(base_stock)

    # Structure 100% alignée GAS
    struct_label, struct_ok = gas_structure_series(
        base_stock,
        w=ALPHA_CFG['PIVOT_W'],
        last_pivots=ALPHA_CFG['STRUCT_LAST_PIVOTS']
    )

    # 3) SCORE (pondérations GAS)
    s_val = pd.Series(0.0, index=base_stock.index)

    s_val += np.where((rsi_gold >= 50) & (rsi_gold <= 70), 15, 0)
    s_val += np.where(v_ratio > 1.5, 25, np.where(v_ratio > 1.1, 12.5, 0))
    s_val += np.where(dist_mm20 <= 0.01, np.where(base_stock['Close'] >= mm20, 20, -10), 0)
    s_val += np.where(struct_ok, 30, 0)
    s_val += np.where(is_sqz, 10, 0)

    # comme GAS: si RS <= 0 => score = 0
    score = pd.Series(np.where(rs_line <= 0, 0, s_val), index=base_stock.index)

    # 4) FILTRE MARCHÉ & ATR D'ENTRÉE
    idx_close_on_stock_dates = base_idx['Close'].reindex(base_stock.index)

    # SMA100
    idx_sma = base_idx['Close'].rolling(ALPHA_CFG['SMA_P'], min_periods=ALPHA_CFG['SMA_P']).mean()
    idx_sma_on_stock_dates = idx_sma.reindex(base_stock.index)

    # pente proche de ta logique existante
    idx_slope = ((idx_sma_on_stock_dates - idx_sma_on_stock_dates.shift(4)) / idx_sma_on_stock_dates.shift(4)).fillna(0)

    # filtre marché strict
    if ALPHA_CFG['MKT_FILTER']:
        mkt_ok = (idx_close_on_stock_dates > idx_sma_on_stock_dates).fillna(False)
    else:
        mkt_ok = pd.Series(True, index=base_stock.index)

    # ATR entrée (même approche que ton v3)
    tr = atr_true_range_series(base_stock)
    atr_vec = tr.rolling(ALPHA_CFG['ATR_P'], min_periods=ALPHA_CFG['ATR_P']).mean().shift(1).fillna(0)

    # 5) DEBUG
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
        'SCORE FINAL': pd.Series(score).round(2)
    })

    try:
        idx_loc = abs(df_debug.index - cible).argmin()
       
