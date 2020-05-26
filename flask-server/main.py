from flask import (Flask, render_template, request, jsonify)
from flask_cors import CORS
import pandas as pd
import pickle
from datetime import date, timedelta
from sklearn import svm 
import numpy as np
from numpy import random


app = Flask("__main__")
CORS(app)

model_list = [['MSFT', 'Microsoft'], ['AAPL', 'Apple'], ['AMZN', 'Amazon'], ['GOOGL', 'Google'], 
['FB', 'Facebook'], ['XOM', 'Exxon Mobil'], ['JNJ', 'Johnson & Johnson'], ['V', 'Visa'],
['PG', 'Procter & Gamble'], ['JPM', 'JPMorgan Chase'], ['UNH', 'UnitedHealth Group'], 
['MA', 'Mastercard'], ['INTC', 'Intel Corporation'], ['VZ', 'Verizon'], ['HD', 'Home Depot'], 
['T', 'AT&T'], ['PFE', 'Pfizer'], ['MRK', 'Merck & Co.'], ['PEP', 'PepsiCo'], 
['DIS', 'Walt Disney Co.']]

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
    #grab a random ticker from the list
    random_index = random.randint(len(model_list))
    random_ticker = model_list[random_index][0]
    model = pickle.load(open(f'model_{random_ticker}.pickle', 'rb'))

    #grab a random date
    days = random.randint(date_range)
    random_start = start + timedelta(days=days)
    random_end = date(random_start.year + 1, random_start.month, random_start.day)

    #load data from server into df for desired timeframe
    X = pd.read_csv('server_stock_data_pct.csv', index_col=0)
    stock_prices = pd.read_csv('server_stock_data.csv', index_col=0)

    series = stock_prices[random_ticker]
    series.index = pd.to_datetime(series.index)
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
    print(series)
    print('tye series: ', type(series))
    series.interpolate(method='linear', inplace=True)
   
    #make a new dataframe and fill with stock data
    columns = ['close', 'predictions']
    df = pd.DataFrame(index=X.index, columns=columns)
    df['close'] = series.values
    df['predictions'] = predictions

    df_json = df.to_json(orient = 'split')

    #Send results in JSON format
    return {'stock': model_list[random_index], 'data': df_json}

    


app.run()
