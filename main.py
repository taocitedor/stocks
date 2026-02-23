from flask import Flask, request, jsonify
import yfinance as yf
import os

app = Flask(__name__)

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

    ticker_list = tickers_string.split(',')
    results = {}

    try:
        # ---------------------------------------------------------
        # MODE SOLO (1 ticker) : On utilise yf.Ticker (100% fiable)
        # ---------------------------------------------------------
        if len(ticker_list) == 1:
            ticker = ticker_list[0]
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="2d")
                
                if df.empty or 'Close' not in df.columns:
                    results[ticker] = {"status": "no_data", "message": "Aucune donnée renvoyée"}
                else:
                    df = df.dropna(subset=['Close'])
                    if not df.empty:
                        last_row = df.iloc[-1]
                        close_val = float(last_row['Close'])
                        
                        change_pct = 0
                        if len(df) >= 2:
                            prev_close = float(df.iloc[-2]['Close'])
                            change_pct = ((close_val - prev_close) / prev_close) * 100

                        results[ticker] = {
                            "date": last_row.name.strftime('%Y-%m-%d'),
                            "close": round(close_val, 3),
                            "variation_veille": round(change_pct, 2),
                            "high": round(float(last_row['High']), 3),
                            "low": round(float(last_row['Low']), 3),
                            "volume": int(last_row['Volume']),
                            "status": "success"
                        }
                    else:
                        results[ticker] = {"status": "no_data"}
            except Exception as e:
                results[ticker] = {"status": "error", "message": str(e)}

        # ---------------------------------------------------------
        # MODE MULTI (>1 ticker) : On utilise yf.download
        # ---------------------------------------------------------
        else:
            data = yf.download(ticker_list, period="2d", group_by='ticker', threads=True, prepost=False)
            for ticker in ticker_list:
                try:
                    df = data[ticker]
                    
                    if df.empty or 'Close' not in df.columns:
                        results[ticker] = {"status": "no_data", "message": "Aucune donnée renvoyée"}
                        continue
                    
                    df = df.dropna(subset=['Close'])
                    if not df.empty:
                        last_row = df.iloc[-1]
                        close_val = float(last_row['Close'])
                        
                        change_pct = 0
                        if len(df) >= 2:
                            prev_close = float(df.iloc[-2]['Close'])
                            change_pct = ((close_val - prev_close) / prev_close) * 100

                        results[ticker] = {
                            "date": last_row.name.strftime('%Y-%m-%d'),
                            "close": round(close_val, 3),
                            "variation_veille": round(change_pct, 2),
                            "high": round(float(last_row['High']), 3),
                            "low": round(float(last_row['Low']), 3),
                            "volume": int(last_row['Volume']),
                            "status": "success"
                        }
                    else:
                        results[ticker] = {"status": "no_data"}
                except Exception as e:
                    results[ticker] = {"status": "error", "message": str(e)}

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

if __name__ == "__main__":
    # Cloud Run utilise le port 8080 par défaut
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
