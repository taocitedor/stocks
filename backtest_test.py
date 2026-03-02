from google.cloud import bigquery
import pandas as pd
import numpy as np


# -------------------
# Fonctions utilitaires
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

    # Protection division par 0
    loss = loss.replace(0, np.nan)

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_score(sub_df):
    if len(sub_df) < 63:
        return 0

    cls = sub_df['Close']
    rsi = RSI(cls).iloc[-1]
    mm20 = cls.tail(20).mean()

    if pd.isna(rsi) or pd.isna(mm20) or mm20 == 0:
        return 0

    score = 0
    score += 15 if 50 <= rsi <= 70 else 0
    score += 20 if abs(cls.iloc[-1] - mm20) / mm20 <= 0.01 else 0

    return score


# -------------------
# Paramètres
# -------------------

PROJECT_ID = "project-16c606d0-6527-4644-907"
DATASET_ID = "Trading"
TABLE_HISTO = "CC_Historique_Cours"

TICKER = "ORA.PA"

VLAB_GLOBAL_SCORE = 86
VLAB_POS_SIZE = 1000
VLAB_FEES = 0.0056
VLAB_ATR_PERIOD = 14


# -------------------
# Fonction principale
# -------------------

def run_backtest_ORA():

    logs = []
    trades = []

    try:

        logs.append("=== START BACKTEST ORA.PA ===")

        # -------------------
        # Connexion BigQuery
        # -------------------
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
                "message": f"Aucune donnée récupérée pour {TICKER}",
                "logs": logs
            }

        logs.append(f"Données récupérées : {len(df)} lignes")

        # -------------------
        # Conversion types SAFE
        # -------------------
        df['Close'] = pd.to_numeric(df['Close'], errors="coerce")
        df['High'] = pd.to_numeric(df['High'], errors="coerce")
        df['Low'] = pd.to_numeric(df['Low'], errors="coerce")
        df['Volume'] = pd.to_numeric(df['Volume'], errors="coerce")
        df['Date'] = pd.to_datetime(df['Date'], errors="coerce")

        df = df.dropna().reset_index(drop=True)

        logs.append(f"Lignes après nettoyage : {len(df)}")

        # -------------------
        # Backtest
        # -------------------
        in_pos = False
        ePrice = 0

        for i in range(63, len(df)):

            curr = df.iloc[i]
            sub_df = df.iloc[:i+1]

            curr_close = curr['Close']

            if pd.isna(curr_close) or curr_close == 0:
                logs.append(f"WARNING: curr_close invalide {curr_close} le {curr['Date']}")
                continue

            atr_series = ATR(sub_df, VLAB_ATR_PERIOD)
            atr = atr_series.iloc[-1]

            if pd.isna(atr) or atr == 0:
                logs.append(f"WARNING: ATR invalide {atr} le {curr['Date']}")
                continue

            vol_pct = atr / curr_close if curr_close != 0 else 0

            score = get_score(sub_df)

            # -------------------
            # ENTRY
            # -------------------
            if not in_pos and score >= VLAB_GLOBAL_SCORE:
                in_pos = True
                ePrice = curr_close
                logs.append(
                    f"ENTRY {curr['Date']} Close={curr_close:.2f} Score={score:.2f}"
                )
                continue

            # -------------------
            # EXIT
            # -------------------
            if in_pos:

                if ePrice == 0:
                    logs.append("ERROR: ePrice = 0")
                    in_pos = False
                    continue

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
                        'EntryPrice': float(ePrice),
                        'ExitPrice': float(curr_close),
                        'Cash': float(trade_cash),
                        'Status': status
                    })

                    logs.append(
                        f"EXIT {curr['Date']} Status={status} Cash={trade_cash:.2f}"
                    )

                    in_pos = False

        # -------------------
        # Résultats globaux
        # -------------------
        nb_gagnes = sum(1 for t in trades if t['Status'] == "GAGNÉ")
        nb_perdus = sum(1 for t in trades if t['Status'] == "PERDU")

        logs.append("=== RÉSULTATS GLOBAUX ===")
        logs.append(f"Nombre de trades : {len(trades)}")
        logs.append(f"GAGNÉS : {nb_gagnes}")
        logs.append(f"PERDUS : {nb_perdus}")

        return {
            "status": "ok",
            "nb_trades": len(trades),
            "nb_gagnes": nb_gagnes,
            "nb_perdus": nb_perdus,
            "trades": trades,
            "logs": logs
        }

    except Exception as e:
        logs.append(f"EXCEPTION: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "logs": logs
        }
