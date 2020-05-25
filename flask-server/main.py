from flask import (Flask, render_template, request, jsonify)
from flask_cors import CORS
import sys
import pandas as pd
import pandas_datareader as pdr
import os
import pickle
from datetime import date, timedelta
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import numpy as np
from numpy import random
from tiingo import TiingoClient

app = Flask("__main__")
CORS(app)
TIINGO_API_KEY = '55bc37569b161cb260a333291d0baf6b3af6d295'
config = {}
config['api_key'] = TIINGO_API_KEY
client = TiingoClient(config)

model_list = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'FB', 'XOM', 'JNJ', 'V', 'PG', 'JPM', 
              'UNH', 'MA', 'INTC', 'VZ', 'HD', 'T', 'PFE', 'MRK', 'PEP', 'DIS']
tickers = pickle.load(open('sp500tickeradjusted.pickle', 'rb'))
start = date(2006, 1, 2)
end = date(2018, 5, 1)
date_range = end - start
date_range = int(date_range.days)

@app.route("/")
def my_index():
    return render_template("index.html")

@app.route('/game')
def return_index():
    return render_template("index.html")


@app.route("/data", methods=['GET'])
def gen_stock_data():
    print('Getting data from server...')
    #grab a random ticker from the list
    random_index = random.randint(len(model_list))
    random_ticker = model_list[random_index]
    model = pickle.load(open(f'model_{random_ticker}.pickle', 'rb'))

    #grab a random date
    days = random.randint(date_range)
    random_start = start + timedelta(days=days)
    random_end = date(random_start.year + 1, random_start.month, random_start.day)

    #get data from Tiingo
    #try:
    #    series = client.get_dataframe(random_ticker,
    #                                frequency='daily',
    #                                metric_name='close', 
    #                                startDate=random_start,
    #                                endDate=random_end)
        
    #except:
    #    e = sys.exc_info()[0]
    #    return f'Error: {e}'

    #load data from server into df for desired timeframe
    X = pd.read_csv('server_stock_data_pct.csv', index_col=0)
    stock_prices = pd.read_csv('server_stock_data.csv', index_col=0)

    series = stock_prices[random_ticker]
    series.index = pd.to_datetime(series.index)
    print('series: ', series)
    series = series.loc[random_start:random_end]
    

    X.index = pd.to_datetime(X.index)
   
    X = X.loc[random_start:random_end]

    #Clean up data
    X = X.replace([np.inf, -np.inf], 0)
    X.fillna(value=0, inplace=True)

    #Extract values and put into model
    X_val = X.values
    predictions = model.predict(X_val)
    #Fill any NaN cells
    series = series.replace([np.inf, -np.inf])
    series.fillna(value=0, inplace=True)
   
    #make a new dataframe and fill with stock data
    columns = ['close', 'predictions']
    df = pd.DataFrame(index=X.index, columns=columns)
    print('CURRENT TICKER: ', random_ticker)
    print('START DATE:', random_start)
    print('END DATE: ', random_end)
    print('X index length: ', len(X.index))
    print('Series Values Length:', len(series.values))
    #df['close'] = series.values
    df['close'] = series.values
    df['predictions'] = predictions

    df_json = df.to_json(orient = 'split')
    #response = {'stock': random_ticker, 'data': df_json}

    #Send results in JSON format
    return {'stock': random_ticker, 'data': df_json}

    


app.run(debug=True)
