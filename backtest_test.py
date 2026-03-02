# backtest_test.py
from google.cloud import bigquery
import pandas as pd
import numpy as np

# -------------------
# Fonctions utilitaires (inchangées)
# -------------------
def SMA(series, period):
    return series.rolling(period).mean()

def ATR(df, period):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_score(sub_df):
    if len(sub_df) < 63:
        return 0
    cls = sub_df['Close']
    rsi = RSI(cls).iloc[-1]
    mm20 = cls.tail(20).mean()
    score = 0
    score += 15 if 50 <= rsi <= 70 else 0
    score += 20 if abs(sub_df['Close'].iloc[-1] - mm20)/mm20 <= 0.01 else 0
    return score

# -------------------
# Fonction principale de backtest pour ORA.PA
# -------------------
def run_backtest_ORA():
    PROJECT_ID = "project-16c606d0-6527-4644-907"
    DATASET_ID = "Trading"
    TABLE_HISTO = "CC_Historique_Cours"
    TICKER = "ORA.PA"
    VLAB_GLOBAL_SCORE = 86
    VLAB_POS_SIZE = 1000
    VLAB_FEES = 0.0056
    VLAB_ATR_PERIOD = 14

    warnings = []
    trades = []

    # Connexion BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT *
    FROM `{DATASET_ID}.{TABLE_HISTO}`
    WHERE Ticker = '{TICKER}'
    ORDER BY Date ASC
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return {"status":"error","message":f"Aucune donnée pour {TICKER}"}

    # Conversion types
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])

    in_pos = False
    ePrice = 0

    for i in range(63, len(df)):
        curr = df.iloc[i]
        sub_df = df.iloc[:i+1]
        curr_close = curr['Close']

        if pd.isna(curr_close) or curr_close == 0:
            warnings.append(f"{curr['Date']} : curr_close invalide ({curr_close})")
            continue

        atr_series = ATR(sub_df.tail(VLAB_ATR_PERIOD))
        atr = atr_series.iloc[-1]
        if pd.isna(atr) or atr == 0:
            warnings.append(f"{curr['Date']} : ATR invalide ({atr})")
            continue

        vol_pct = atr / curr_close

        score = get_score(sub_df)

        # Ouverture position
        if not in_pos and score >= VLAB_GLOBAL_SCORE:
            in_pos = True
            ePrice = curr_close
            warnings.append(f"{curr['Date']} : Ouverture position Close={curr_close} Score={score}")
            continue

        # Fermeture position (TP/SL simplifié)
        if in_pos:
            high_perf = (curr['High'] - ePrice) / ePrice
            low_perf = (curr['Low'] - ePrice) / ePrice
            hitTP = high_perf >= 0.05
            hitSL = low_perf <= -0.05
            if hitTP or hitSL:
                status = "GAGNÉ" if hitTP else "PERDU"
                trade_cash = (curr_close - ePrice - VLAB_FEES) * VLAB_POS_SIZE
                trades.append({
                    'Date': str(curr['Date']),
                    'Ticker': TICKER,
                    'EntryPrice': ePrice,
                    'ExitPrice': curr_close,
                    'Cash': trade_cash,
                    'Status': status
                })
                warnings.append(f"{curr['Date']} : Fermeture position Status={status} Cash={trade_cash:.2f}")
                in_pos = False

    # Résultats globaux
    nb_gagnes = sum(1 for t in trades if t['Status']=="GAGNÉ")
    nb_perdus = sum(1 for t in trades if t['Status']=="PERDU")

    return {
        "status":"ok",
        "trades": trades,
        "nb_gagnes": nb_gagnes,
        "nb_perdus": nb_perdus,
        "warnings": warnings
    }
