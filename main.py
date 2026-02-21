from flask import Flask, request, jsonify
import yfinance as yf
import os

app = Flask(__name__)

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

@app.route('/get_batch_data', methods=['GET'])
def get_batch_data():
    tickers_string = request.args.get('tickers')
    if not tickers_string:
        return jsonify({"error": "Aucun ticker fourni"}), 400

    ticker_list = tickers_string.split(',')
    results = {}

    try:
        # On télécharge 2 jours pour tout le monde
        data = yf.download(ticker_list, period="2d", group_by='ticker', threads=False, prepost=False)
        
        for ticker in ticker_list:
            try:
                # Extraction des lignes selon le format (MultiIndex ou Series)
                df = data if len(ticker_list) == 1 else data[ticker]
                df = df.dropna()
                
                if len(df) >= 1:
                    last_row = df.iloc[-1]
                    change_pct = 0
                    if len(df) >= 2:
                        prev_close = df.iloc[-2]['Close']
                        change_pct = ((last_row['Close'] - prev_close) / prev_close) * 100

                    results[ticker] = {
                        "date": last_row.name.strftime('%Y-%m-%d'),
                        "close": round(last_row['Close'], 3),
                        "variation_veille": round(change_pct, 2),
                        "high": round(last_row['High'], 3),
                        "low": round(last_row['Low'], 3),
                        "volume": int(last_row['Volume']),
                        "status": "success"
                    }
                else:
                    results[ticker] = {"status": "no_data"}
            except:
                results[ticker] = {"status": "error"}

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
