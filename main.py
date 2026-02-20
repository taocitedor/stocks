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
        # On demande l'historique du jour
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

@app.route('/get_batch_data', methods=['GET'])
def get_batch_data():
    # On attend une liste séparée par des virgules (ex: AI.PA,AAPL,MC.PA)
    tickers_string = request.args.get('tickers')
    if not tickers_string:
        return jsonify({"error": "Aucun ticker fourni"}), 400

    ticker_list = tickers_string.split(',')
    results = {}

    try:
        # Téléchargement groupé (plus rapide)
        data = yf.download(ticker_list, period="1d", group_by='ticker', threads=True)
        
        for ticker in ticker_list:
            try:
                # Gestion du cas où yf.download renvoie un format différent pour 1 seul ticker
                if len(ticker_list) == 1:
                    last_row = data.iloc[-1]
                else:
                    last_row = data[ticker].iloc[-1]

                results[ticker] = {
                    "close": round(last_row['Close'], 3),
                    "high": round(last_row['High'], 3),
                    "low": round(last_row['Low'], 3),
                    "volume": int(last_row['Volume']),
                    "status": "success"
                }
            except:
                results[ticker] = {"status": "error"}

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
