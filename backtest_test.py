from google.cloud import bigquery
import pandas as pd
import numpy as np


# ==============================
# INDICATEURS
# ==============================

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
    loss = loss.replace(0, np.nan)  # protection division 0
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ==============================
# CONFIG
# ==============================

PROJECT_ID = "project-16c606d0-6527-4644-907"
DATASET_ID = "Trading"
TABLE_HISTO = "CC_Historique_Cours"

TICKER = "ORA.PA"

VLAB_GLOBAL_SCORE = 86
VLAB_POS_SIZE = 1000
VLAB_FEES = 0.0056
VLAB_ATR_PERIOD = 14


# ==============================
# BACKTEST DEBUG COMPLET
# ==============================

def run_backtest_ORA():

    logs = []
    trades = []

    try:

        logs.append("=== START BACKTEST ORA.PA ===")

        # ----------------------
        # BigQuery
        # ----------------------
        client = bigquery.Client(project=PROJECT_ID)

        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_HISTO}`
        WHERE Ticker = '{TICKER}'
        ORDER BY Date ASC
        """

        df = client.query(query).to_dataframe()

        if df.empty:
            return {
                "status": "error",
                "message": "Aucune donnée récupérée",
                "logs": logs
            }

        logs.append(f"Données récupérées : {len(df)} lignes")

        # ----------------------
        # Nettoyage sécurisé
        # ----------------------
        df['Close'] = pd.to_numeric(df['Close'], errors="coerce")
        df['High'] = pd.to_numeric(df['High'], errors="coerce")
        df['Low'] = pd.to_numeric(df['Low'], errors="coerce")
        df['Volume'] = pd.to_numeric(df['Volume'], errors="coerce")
        df['Date'] = pd.to_datetime(df['Date'], errors="coerce")

        df = df.dropna().reset_index(drop=True)

        logs.append(f"Lignes après nettoyage : {len(df)}")

        # ----------------------
        # ANALYSE STATISTIQUE
        # ----------------------
        logs.append("=== ANALYSE STATISTIQUE ===")

        df["RSI"] = RSI(df["Close"])
        df["MM20"] = df["Close"].rolling(20).mean()
        df["dist_mm20_pct"] = abs(df["Close"] - df["MM20"]) / df["MM20"]

        logs.append(f"RSI min : {round(df['RSI'].min(),2)}")
        logs.append(f"RSI max : {round(df['RSI'].max(),2)}")
        logs.append(f"RSI moyen : {round(df['RSI'].mean(),2)}")

        logs.append(f"Distance MM20 moyenne : {round(df['dist_mm20_pct'].mean(),4)}")
        logs.append(f"Distance MM20 max : {round(df['dist_mm20_pct'].max(),4)}")

        logs.append(f"Close min : {df['Close'].min()}")
        logs.append(f"Close max : {df['Close'].max()}")

        logs.append(f"Nombre Close = 0 : {(df['Close']==0).sum()}")

        # ----------------------
        # BOUCLE DEBUG SCORE
        # ----------------------
        max_score = 0
        score_values = []

        for i in range(63, len(df)):

            curr = df.iloc[i]
            sub_df = df.iloc[:i+1]

            curr_close = curr['Close']
            if pd.isna(curr_close) or curr_close == 0:
                continue

            rsi = RSI(sub_df["Close"]).iloc[-1]
            mm20 = sub_df["Close"].tail(20).mean()

            if pd.isna(rsi) or pd.isna(mm20) or mm20 == 0:
                continue

            score = 0

            rsi_ok = 50 <= rsi <= 70
            prox_ok = abs(curr_close - mm20)/mm20 <= 0.01

            if rsi_ok:
                score += 15
            if prox_ok:
                score += 20

            score_values.append(score)

            if score > max_score:
                max_score = score
                logs.append(
                    f"NOUVEAU MAX SCORE {score} "
                    f"Date={curr['Date']} RSI={round(rsi,2)} "
                    f"DistMM20={round(abs(curr_close-mm20)/mm20,4)}"
                )

        # ----------------------
        # STAT SCORE
        # ----------------------
        logs.append("=== STAT SCORE ===")

        if score_values:
            logs.append(f"Score max : {max_score}")
            logs.append(f"Score moyen : {round(np.mean(score_values),2)}")
            logs.append(f"Score > 0 : {sum(1 for s in score_values if s > 0)}")
            logs.append(f"Score = 35 : {sum(1 for s in score_values if s == 35)}")

        logs.append(f"Seuil entrée : {VLAB_GLOBAL_SCORE}")

        return {
            "status": "ok",
            "nb_trades": len(trades),
            "logs": logs
        }

    except Exception as e:
        logs.append(f"EXCEPTION: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "logs": logs
        }
