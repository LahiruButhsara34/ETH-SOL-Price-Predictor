from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

app = Flask(__name__)

def train_and_predict_future(ticker, target_date_str):
    # 1. Automatically download historical cryptocurrency data
    start_date = '2018-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or len(df) < 100:
        raise Exception("Insufficient data available for training.")

    # Fetch Closing Prices
    data = df[['Close']].values.reshape(-1,1)
    dates = [d.strftime('%Y-%m-%d') for d in df.index]

    # 2. Scale Data (Between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 3. Create Sequences with 90 Time-Steps
    time_steps = 90
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 4. Build Deep LSTM Model
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

  
    # Define Callbacks to get the model with the minimum validation loss
    model_filename = 'best_crypto_model.keras'
    checkpoint = ModelCheckpoint(
        model_filename,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min'
    )

    # Train with Validation Split (Time-series data shuffle = False)
    model.fit(
        X, y,
        epochs=30,
        batch_size=64,
        validation_split=0.15,
        shuffle=False,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    # Load the absolute best performing model based on val_loss
    if os.path.exists(model_filename):
        model = load_model(model_filename)

    # 5. Calculate total number of days up to the target date
    last_date = df.index[-1].to_pydatetime()
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    
    days_to_predict = (target_date - last_date).days

    if days_to_predict <= 0:
        raise Exception("Please select a date after today.")

    # Iterative Multi-Step Forecasting
    current_input = scaled_data[-time_steps:].copy()
    future_predictions = []
    future_dates = []

    for i in range(1, days_to_predict + 1):
        x_input = current_input.reshape(1, time_steps, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        
        # Shift input data for the next step prediction
        current_input = np.append(current_input[1:], [[pred]], axis=0)

        # Sample points monthly or on the final target date for charting
        if i % 30 == 0 or i == days_to_predict:
            pred_unscaled = scaler.inverse_transform([[pred]])[0][0]
            future_predictions.append(round(float(pred_unscaled), 2))
            next_date = last_date + timedelta(days=i)
            future_dates.append(next_date.strftime('%Y-%m-%d'))

    # Select recent 100 historical records
    historical_subset_prices = [round(float(x[0]), 2) for x in data[-100:]]
    historical_subset_dates = dates[-100:]

    latest_actual_price = round(float(data[-1][0]), 2)
    final_predicted_price = future_predictions[-1]

    return {
        "ticker": ticker,
        "latest_price": latest_actual_price,
        "final_predicted_price": final_predicted_price,
        "hist_dates": historical_subset_dates,
        "hist_prices": historical_subset_prices,
        "future_dates": future_dates,
        "future_prices": future_predictions
    }


@app.route('/')
def index():
    today = datetime.now().strftime('%Y-%m-%d')
    max_date = (datetime.now() + timedelta(days=4*365)).strftime('%Y-%m-%d')
    return render_template('coin_details.html', today=today, max_date=max_date)


@app.route('/getresults', methods=['POST'])
def getresults():
    if request.method == 'POST':
        try:
            ticker = request.form.get('coin_name', 'ETH-USD')
            target_date = request.form.get('target_date')

            today_str = datetime.now().strftime('%Y-%m-%d')
            max_date_str = (datetime.now() + timedelta(days=4*365)).strftime('%Y-%m-%d')

            if not target_date:
                return render_template('coin_details.html', error="Please select a target date.", today=today_str, max_date=max_date_str)
            
            if target_date > max_date_str or target_date <= today_str:
                return render_template('coin_details.html', error="Please select a date between today and the next 4 years.", today=today_str, max_date=max_date_str)

            data_res = train_and_predict_future(ticker, target_date)
            
            price_change = data_res['final_predicted_price'] - data_res['latest_price']
            change_percent = (price_change / data_res['latest_price']) * 100

            results = {
                "coin": "Ethereum (ETH)" if ticker == 'ETH-USD' else "Solana (SOL)",
                "ticker": ticker,
                "target_date": data_res['future_dates'][-1],
                "latest_price": data_res['latest_price'],
                "predicted_price": data_res['final_predicted_price'],
                "change": round(price_change, 2),
                "change_percent": round(change_percent, 2),
                "hist_dates": data_res['hist_dates'],
                "hist_prices": data_res['hist_prices'],
                "future_dates": data_res['future_dates'],
                "future_prices": data_res['future_prices']
            }

            return render_template('coin_results.html', results=results)

        except Exception as e:
            today_str = datetime.now().strftime('%Y-%m-%d')
            max_date_str = (datetime.now() + timedelta(days=5*365)).strftime('%Y-%m-%d')
            return render_template('coin_details.html', error=f"An error occurred: {str(e)}", today=today_str, max_date=max_date_str)


if __name__ == '__main__':
   # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    