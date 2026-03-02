# backtest_test.py
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------
# FONCTIONS UTILITAIRES
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
    """Reconstruit le scoring VLAB Alpha Gold Hybrid Guard"""
    if len(sub_df) < 63:
        return 0
    cls = sub_df['Close']
    rsi = RSI(cls).iloc[-1]
    mm20 = cls.tail(20).mean()
    vRatio = sub_df['Volume'].iloc[-1] / (sub_df['Volume'].tail(20).mean() or 1)
    
    # Structure simplifiée (HH+HL)
    highs = sub_df['High'].tail(20)
    lows = sub_df['Low'].tail(20)
    struct_score = 30 if (highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]) else 0
    
    # Squeeze simplifié
    sma20 = cls.tail(20).mean()
    sd = cls.tail(20).std()
    atr20 = (sub_df['High'].tail(20) - sub_df['Low'].tail(20)).mean()
    sqz = 10 if (sma20 + 2*sd < sma20 + 1.5*atr20 and sma20 - 2*sd > sma20 - 1.5*atr20) else 0

    score = 0
    score += 15 if 50 <= rsi <= 70 else 0
    score += 20 if abs(cls.iloc[-1] - mm20)/mm20 <= 0.01 else 0
    score += 25 if vRatio > 1.5 else (12.5 if vRatio > 1.1 else 0)
    score += struct_score
    score += sqz
    return score

# -------------------
# FONCTION PRINCIPALE BACKTEST ORA
# -------------------
def run_backtest_ORA(date_debug=None):
    logs = []
    trades = []

    PROJECT_ID = "project-16c606d0-6527-4644-907"
    DATASET_ID = "Trading"
    TABLE_HISTO = "CC_Historique_Cours"
    TICKER = "ORA.PA"
    VLAB_GLOBAL_SCORE = 86
    VLAB_POS_SIZE = 1000
    VLAB_FEES = 0.0056
    VLAB_ATR_PERIOD = 14
    VLAB_GLOBAL_SL = 0.10

    logs.append(f"=== START BACKTEST {TICKER} ===")

    # -------------------
    # Connexion BigQuery
    # -------------------
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT *
        FROM `{DATASET_ID}.{TABLE_HISTO}`
        WHERE Ticker = '{TICKER}'
        ORDER BY Date ASC
    """
    df = client.query(query).to_dataframe()

    if df.empty:
        logs.append(f"Erreur : aucune donnée récupérée pour {TICKER}")
        return {"status":"error","message": "aucune donnée", "logs": logs}

    # Conversion types
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    logs.append(f"Données récupérées : {len(df)} lignes pour {TICKER}")
    logs.append(f"Lignes après nettoyage : {len(df)}")

    # -------------------
    # Statistiques rapides
    # -------------------
    cls = df['Close']
    rsi_series = RSI(cls)
    logs.append("=== ANALYSE STATISTIQUE ===")
    logs.append(f"RSI min : {rsi_series.min():.2f}")
    logs.append(f"RSI max : {rsi_series.max():.2f}")
    logs.append(f"RSI moyen : {rsi_series.mean():.2f}")
    mm20 = cls.rolling(20).mean()
    logs.append(f"Distance MM20 moyenne : {((cls - mm20)/mm20).abs().mean():.4f}")
    logs.append(f"Distance MM20 max : {((cls - mm20)/mm20).abs().max():.4f}")
    logs.append(f"Close min : {cls.min():.3f}")
    logs.append(f"Close max : {cls.max():.3f}")
    logs.append(f"Nombre Close = 0 : {(cls==0).sum()}")

    # -------------------
    # Boucle de backtest
    # -------------------
    in_pos = False
    ePrice = 0
    new_max_score = 0

    for i in range(63, len(df)):
        curr = df.iloc[i]
        sub_df = df.iloc[:i+1]

        # Vérification prix valide
        if pd.isna(curr['Close']) or curr['Close']==0:
            logs.append(f"WARNING: curr_close invalide {curr['Close']} le {curr['Date']}")
            continue

        # ATR
        atr_series = ATR(sub_df.tail(VLAB_ATR_PERIOD), VLAB_ATR_PERIOD)
        atr = atr_series.iloc[-1]
        if pd.isna(atr) or atr == 0:
            logs.append(f"WARNING: ATR invalide {atr} le {curr['Date']}")
            continue
        vol_pct = atr / curr['Close']

        # Score
        score = get_score(sub_df)
        if score > new_max_score:
            new_max_score = score
            logs.append(f"NOUVEAU MAX SCORE {score:.0f} Date={curr['Date'].date()}")

        # Ouverture position
        if not in_pos and score >= VLAB_GLOBAL_SCORE:
            in_pos = True
            ePrice = curr['Close']
            logs.append(f"Ouverture position {TICKER} le {curr['Date'].date()} Close={curr['Close']:.2f} Score={score:.2f}")
            continue

        # Fermeture position TP/SL simplifié
        if in_pos:
            high_perf = (curr['High'] - ePrice) / ePrice
            low_perf = (curr['Low'] - ePrice) / ePrice
            hitTP = high_perf >= 0.05
            hitSL = low_perf <= -VLAB_GLOBAL_SL
            if hitTP or hitSL:
                status = "GAGNÉ" if hitTP else "PERDU"
                trade_cash = (curr['Close'] - ePrice - VLAB_FEES) * VLAB_POS_SIZE
                trades.append({
                    'Date': curr['Date'],
                    'Ticker': TICKER,
                    'EntryPrice': ePrice,
                    'ExitPrice': curr['Close'],
                    'Cash': trade_cash,
                    'Status': status
                })
                logs.append(f"Fermeture position {TICKER} le {curr['Date'].date()} Status={status} Cash={trade_cash:.2f}")
                in_pos = False

    # -------------------
    # Résultats globaux
    # -------------------
    nb_gagnes = sum(1 for t in trades if t['Status']=="GAGNÉ")
    nb_perdus = sum(1 for t in trades if t['Status']=="PERDU")
    logs.append("\n=== RÉSULTATS GLOBAUX ===")
    logs.append(f"Nombre de trades : {len(trades)}")
    logs.append(f"GAGNÉS : {nb_gagnes}")
    logs.append(f"PERDUS : {nb_perdus}")

    return {
        "logs": logs,
        "nb_trades": len(trades),
        "nb_gagnes": nb_gagnes,
        "nb_perdus": nb_perdus,
        "trades": trades,
        "status":"ok"
    }

# -------------------------
# DEBUG LOCAL
# -------------------------
if __name__ == "__main__":
    # Test sur une date historique
    result = run_backtest_ORA("2024-05-31")
    for l in result["logs"]:
        print(l)
