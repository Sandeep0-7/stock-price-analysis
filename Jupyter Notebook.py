{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Stock Price Analysis\n",
    "Time series forecasting with ARIMA, LSTM, and Prophet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import yfinance as yf\n",
    "from prophet import Prophet\n",
    "# [Rest of your imports]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run analysis\n",
    "stock_prices = yf.download('AAPL', start='2018-01-01', end='2023-12-31')['Close']\n",
    "stock_prices.plot(title='AAPL Stock Price')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 }
}