from flask import Flask, request, jsonify
from google.cloud import bigquery
import yfinance as yf
import os
import backtest_test
import sigma

app = Flask(__name__)

@app.route("/run_test2", methods=["GET"])
def run_test2():
    try:
        result = sigma.alpha_engine_v3()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)})


@app.route("/run_test", methods=["GET"])
def run_test():
    try:
        result = backtest_test.run_vlab_backtest_full()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)})


@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return jsonify({"status": "ok"})

@app.route('/test_bq', methods=['GET'])
def test_bq():
    project_id = request.args.get('project')
    dataset_id = request.args.get('dataset')
    table_id = request.args.get('table')

    if not all([project_id, dataset_id, table_id]):
        return jsonify({"error": "project, dataset, table requis"}), 400

    try:
        client = bigquery.Client(project=project_id)
        query = f"SELECT * FROM `{dataset_id}.{table_id}` LIMIT 5"
        df = client.query(query).to_dataframe()

        # logs pour debug
        print("=== BigQuery HEAD ===")
        print(df.head())
        print(df.dtypes)

        return jsonify({
            "status": "ok",
            "rows_returned": len(df),
            "columns": df.columns.tolist(),
            "sample_data": df.head(3).to_dict(orient="records")
        })
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500
        

@app.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    ticker_symbol = request.args.get('ticker')
    if not ticker_symbol:
        return jsonify({"error": "Ticker manquant"}), 400

    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1d")
        
        if hist.empty:
            return jsonify({"error": "Donnees introuvables pour ce ticker"}), 404

        last_row = hist.iloc[-1]
        
        return jsonify({
            "ticker": ticker_symbol,
            "date": last_row.name.strftime('%Y-%m-%d'),
            "close": round(last_row['Close'], 3),
            "high": round(last_row['High'], 3),
            "low": round(last_row['Low'], 3),
            "volume": int(last_row['Volume']),
            "currency": stock.fast_info.get('currency', 'EUR'),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/get_batch_data', methods=['GET'])
def get_batch_data():
    tickers_string = request.args.get('tickers')
    if not tickers_string:
        return jsonify({"error": "Aucun ticker fourni"}), 400

    ticker_list = [t.strip() for t in tickers_string.split(',')]
    results = {}

    try:
        # On utilise yf.download pour TOUT (Solo ou Multi) avec group_by='ticker'
        # Cela garantit une structure de données identique.
        data = yf.download(ticker_list, period="5d", group_by='ticker', threads=True, auto_adjust=False)
        
        for ticker in ticker_list:
            try:
                # Extraction du DataFrame selon que c'est du multi-index ou non
                if len(ticker_list) > 1:
                    df = data[ticker].dropna(subset=['Close'])
                else:
                    df = data.dropna(subset=['Close'])

                if df.empty or len(df) < 2:
                    results[ticker] = {"status": "no_data", "message": "Pas assez d'historique (min 2j)"}
                    continue
                
                # Appel de la logique de calcul sécurisée
                results[ticker] = process_ticker_logic(ticker, df)
                
            except Exception as e:
                results[ticker] = {"status": "error", "message": str(e)}

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_ticker_logic(ticker, df):
    # 1. Identification des lignes
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # 2. Récupération des dates pour vérifier le D-1 vs D-2
    date_cloture = last_row.name.strftime('%Y-%m-%d')
    date_veille = prev_row.name.strftime('%Y-%m-%d')
    
    close_val = float(last_row['Close'])
    prev_close = float(prev_row['Close'])
    
    # 3. Calcul de la variation
    change_pct = ((close_val - prev_close) / prev_close) * 100
    source = "Historical"

    # 4. Sécurité Anti-Stale (Si le marché est ouvert mais que le prix n'a pas bougé)
    # Note : On ne déclenche le Live que si Volume > 0 (marché ouvert)
    if close_val == prev_close and last_row['Volume'] > 0:
        t_obj = yf.Ticker(ticker)
        df_live = t_obj.history(period="1d", interval="1m")
        if not df_live.empty:
            live_price = float(df_live.iloc[-1]['Close'])
            if live_price != close_val:
                close_val = live_price
                change_pct = ((close_val - prev_close) / prev_close) * 100
                source = "Live_1m_Force"

    # 5. Gestion des statuts (Filtre 10% devient un WARNING, pas une erreur)
    status = "success"
    message = ""
    if abs(change_pct) > 10:
        status = "warning_coherence"
        message = f"Variation forte ({round(change_pct, 2)}%) détectée."
    elif close_val == prev_close:
        status = "stale_data"
        message = "Prix identique à la veille."

    return {
        "date": date_cloture,
        "date_comparaison": date_veille, # Pour vérifier si on compare bien aujourd'hui vs hier
        "close": round(close_val, 3),
        "variation_veille": round(change_pct, 2),
        "high": round(float(last_row['High']), 3),
        "low": round(float(last_row['Low']), 3),
        "volume": int(last_row['Volume']),
        "status": status,
        "message": message,
        "source_type": source
    }
    
@app.route('/get_historic_data', methods=['GET'])
def get_historic_data():
    ticker_symbol = request.args.get('ticker')
    target_date = request.args.get('date')  # Format attendu: YYYY-MM-DD
    
    if not ticker_symbol or not target_date:
        return jsonify({"error": "Ticker et date (YYYY-MM-DD) requis"}), 400

    try:
        stock = yf.Ticker(ticker_symbol)
        # On récupère une petite fenêtre autour de la date pour être sûr d'avoir la donnée
        # (car le marché est fermé le week-end)
        df = stock.history(start=target_date, end=None, period="1d")
        
        if df.empty:
            return jsonify({"error": f"Aucune cotation pour le {target_date} (marché fermé ?)"}), 404

        # On prend la ligne qui correspond exactement ou la plus proche
        last_row = df.iloc[0]
        
        return jsonify({
            "ticker": ticker_symbol,
            "requested_date": target_date,
            "actual_date": last_row.name.strftime('%Y-%m-%d'),
            "close": round(last_row['Close'], 3),
            "high": round(last_row['High'], 3),
            "low": round(last_row['Low'], 3),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
        
@app.route('/get_range_data', methods=['GET'])
def get_range_data():
    ticker_symbol = request.args.get('ticker')
    start_date = request.args.get('start')  # Format: YYYY-MM-DD
    end_date = request.args.get('end')      # Format: YYYY-MM-DD
    
    if not all([ticker_symbol, start_date, end_date]):
        return jsonify({"error": "Paramètres manquants : ticker, start, end"}), 400

    try:
        stock = yf.Ticker(ticker_symbol)
        # Note : end_date dans yfinance est exclusif, on récupère donc jusqu'à la veille de end_date
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            return jsonify({"error": "Aucune donnée sur cette période", "status": "no_data"}), 404

        # Transformation du DataFrame en liste de dictionnaires
        history_list = []
        for index, row in df.iterrows():
            history_list.append({
                "date": index.strftime('%Y-%m-%d'),
                "close": round(float(row['Close']), 3),
                "high": round(float(row['High']), 3),
                "low": round(float(row['Low']), 3),
                "volume": int(row['Volume'])
            })
        
        return jsonify({
            "ticker": ticker_symbol,
            "start": start_date,
            "end": end_date,
            "count": len(history_list),
            "history": history_list,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/get_batch_historic', methods=['GET'])
def get_batch_historic():
    tickers_string = request.args.get('tickers')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not all([tickers_string, start_date, end_date]):
        return jsonify({"error": "Paramètres manquants : tickers, start, end"}), 400

    ticker_list = [t.strip() for t in tickers_string.split(',')]
    results = {}

    try:
        # threads=True pour garder ton BIT bas et ta vitesse haute
        data = yf.download(ticker_list, start=start_date, end=end_date, group_by='ticker', threads=True)
        
        for ticker in ticker_list:
            try:
                # Récupération du DataFrame pour le ticker
                if len(ticker_list) == 1:
                    df = data
                else:
                    df = data[ticker]
                
                df = df.dropna(subset=['Close'])
                
                if df.empty:
                    results[ticker] = {"status": "no_data", "ticker": ticker}
                    continue

                history = []
                for index, row in df.iterrows():
                    history.append({
                        "date": index.strftime('%Y-%m-%d'),
                        "open": round(float(row['Open']), 3),
                        "high": round(float(row['High']), 3),
                        "low": round(float(row['Low']), 3),
                        "close": round(float(row['Close']), 3),
                        "volume": int(row['Volume'])
                    })
                
                results[ticker] = {
                    "ticker": ticker,
                    "status": "success",
                    "history": history
                }
            except Exception as e:
                results[ticker] = {"ticker": ticker, "status": "error", "message": str(e)}

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Cloud Run utilise le port 8080 par défaut
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
