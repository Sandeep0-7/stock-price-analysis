import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
from datetime import datetime, timedelta

# 1. Data Collection
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# 2. Visualization
def plot_stock_prices(series, title):
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.show()

# 3. Time Series Decomposition
def decompose_time_series(series, period=252):  # 252 trading days in a year
    result = seasonal_decompose(series, model='additive', period=period)
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(result.observed)
    plt.title('Observed')

    plt.subplot(4, 1, 2)
    plt.plot(result.trend)
    plt.title('Trend')

    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal)
    plt.title('Seasonal')

    plt.subplot(4, 1, 4)
    plt.plot(result.resid)
    plt.title('Residual')

    plt.tight_layout()
    plt.show()
    return result

# 4. Volatility Analysis
def analyze_volatility(series):
    returns = np.log(series / series.shift(1)).dropna()

    # Plot returns
    plt.figure(figsize=(12, 6))
    plt.plot(returns)
    plt.title('Daily Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.show()

    # Plot volatility (rolling standard deviation)
    rolling_volatility = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_volatility)
    plt.title('Rolling Volatility (21-day, annualized)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

    print(f"Average Daily Return: {float(returns.mean()):.4f}")
    print(f"Daily Volatility (std dev): {float(returns.std()):.4f}")
    print(f"Annualized Volatility: {float(returns.std() * np.sqrt(252)):.4f}")

    return returns

# 5. Stationarity Check
def check_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    return result

# 5a. Autocorrelation and Partial Autocorrelation
def plot_acf_pacf(series):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plot_acf(series, ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')

    plt.subplot(2, 1, 2)
    plot_pacf(series, ax=plt.gca(), lags=40)
    plt.title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()

# 6. ARIMA Model
def arima_forecast(series, test_size=0.2):
    # Split data
    train_size = int(len(series) * (1 - test_size))
    train, test = series[:train_size], series[train_size:]

    # Fit ARIMA model
    model = ARIMA(train, order=(5,1,0))  # Example order, should be optimized
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test, forecast))
    print(f'ARIMA RMSE: {rmse:.2f}')

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='ARIMA Forecast')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.show()

    return rmse

# 7. LSTM Model
def lstm_forecast(series, n_steps=60, n_features=1, epochs=50, batch_size=32):
    # Prepare data
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back)]
            X.append(a)
            Y.append(dataset[i + look_back])
        return np.array(X), np.array(Y)

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(series.values.reshape(-1, 1))

    # Split data
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[:train_size], dataset[train_size:]

    # Reshape into X=t and Y=t+1
    look_back = n_steps
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, n_features)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, test_predict))
    print(f'LSTM RMSE: {rmse:.2f}')

    # Plot
    plt.figure(figsize=(12, 6))

    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = test_predict

    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), label='Actual')
    plt.plot(train_predict_plot, label='Train Predict')
    plt.plot(test_predict_plot, label='Test Predict')
    plt.title('LSTM Forecast')
    plt.legend()
    plt.show()

    return rmse

# 8. Prophet Model
def prophet_forecast(series, test_size=0.2):
    # Prepare data for Prophet
    df = series.reset_index()
    df.columns = ['ds', 'y']

    # Split data
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]

    # Fit model
    model = Prophet()
    model.fit(train)

    # Make future dataframe
    future = model.make_future_dataframe(periods=len(test))

    # Forecast
    forecast = model.predict(future)

    # Calculate RMSE
    y_true = test['y'].values
    y_pred = forecast['yhat'].values[-len(test):]
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f'Prophet RMSE: {rmse:.2f}')

    # Plot
    fig1 = model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.show()

    fig2 = model.plot_components(forecast)
    plt.show()

    return rmse

# Main Analysis
def main():
    # Parameters
    ticker = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2023-12-31'

    # 1. Download data
    print("Downloading stock data...")
    stock_prices = download_stock_data(ticker, start_date, end_date)

    # 2. Visualize data
    print("\nVisualizing stock prices...")
    plot_stock_prices(stock_prices, f'{ticker} Stock Price (2018-2023)')

    # 3. Decompose time series
    print("\nDecomposing time series...")
    decomposition = decompose_time_series(stock_prices)

    # 4. Analyze volatility
    print("\nAnalyzing volatility...")
    returns = analyze_volatility(stock_prices)

    # 5. Check stationarity
    print("\nChecking stationarity of returns...")
    check_stationarity(returns)

    # 5a. Plot ACF and PACF
    print("\nPlotting ACF and PACF...")
    plot_acf_pacf(stock_prices)

    # Add the pie chart code here
    print("\nPlotting Pie Chart of Daily Returns...")
    # Calculate positive, negative, and zero returns
    positive_returns = (returns > 0).sum()
    negative_returns = (returns < 0).sum()
    zero_returns = (returns == 0).sum()

    # Create data for the pie chart
    labels = ['Positive Returns', 'Negative Returns', 'Zero Returns']
    # Extract scalar values from the pandas Series
    sizes = np.array([positive_returns.iloc[0], negative_returns.iloc[0], zero_returns.iloc[0]])
    colors = ['lightgreen', 'salmon', 'lightblue']
    explode = (0.1, 0, 0)  # explode the 'Positive Returns' slice

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Proportion of Daily Stock Returns (Positive vs. Negative)')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


    # 6. ARIMA forecasting
    print("\nRunning ARIMA forecast...")
    arima_rmse = arima_forecast(stock_prices)

    # 7. LSTM forecasting
    print("\nRunning LSTM forecast...")
    lstm_rmse = lstm_forecast(stock_prices)

    # 8. Prophet forecasting
    print("\nRunning Prophet forecast...")
    prophet_rmse = prophet_forecast(stock_prices)

    # Compare models
    print("\nModel Comparison:")
    print(f"ARIMA RMSE: {arima_rmse:.2f}")
    print(f"LSTM RMSE: {lstm_rmse:.2f}")
    print(f"Prophet RMSE: {prophet_rmse:.2f}")

if __name__ == "__main__":
    main()