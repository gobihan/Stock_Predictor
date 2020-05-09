from django.http import HttpResponse, JsonResponse
import tensorflow as tf
# print(tf.__version__)
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import keras
import os 
import datetime
import json
# import schedule
# import time
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from math import sqrt


from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from joblib import dump, load
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
# scaler = MinMaxScaler()

def job():
    print("I'm working...")

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def add_stock(request, stock_code):
    train_models(stock_code, True)
    return JsonResponse({'message':"Successfully added stock"})

def find_stock(request, stock_code):

    ticker = yf.Ticker(stock_code)

    print(ticker.info)

    return JsonResponse({"name":ticker.info['shortName'], "sector": ticker.info['sector'], "stock_code": str.upper(stock_code)})


def getSdChangeInPrice(close_prices, lookback = 20):
    daily_vol = close_prices.index.searchsorted(close_prices.index - pd.Timedelta(days=1))
    daily_vol = daily_vol[daily_vol>0]
    daily_vol = pd.Series(data=close_prices.index[daily_vol - 1], index=close_prices.index[close_prices.shape[0]-daily_vol.shape[0]:])

    try:
        daily_vol = (close_prices.loc[daily_vol.index] / close_prices.loc[daily_vol.values].values-1) # daily returns
    except Exception as e:
        print('error: {}\nplease confirm no duplicate indices'.format(type(e).__name__, e))

    daily_vol = daily_vol.ewm(span=lookback).std()

    return daily_vol.iloc[1:]

def recommendation_case_study(stock_code):

    lr_buy = tf.keras.models.load_model(dir_path+'/LR_Recommender_Buy.h5')
    lr_sell = tf.keras.models.load_model(dir_path+'/LR_Recommender_Sell.h5')
    lstm_buy = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Buy.h5')
    lstm_sell = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Sell.h5')

    stock = yf.Ticker(stock_code)
    # df = stock.history(period='7y')
    df = yf.download(stock_code, start='2012-01-01', end='2019-12-17')
    # df.drop(["Dividends", "Stock Splits"], axis = 1, inplace = True) 
    df.drop(["Adj Close"], axis = 1, inplace = True) 

    labelled_stock_data = TripleBarrierMethod(df)
    labelled_stock_data = labelled_stock_data.dropna()

    print(labelled_stock_data)

    df = labelled_stock_data.copy()
    
    target_column = ['Buy'] 
    target_column1 = ['Sell']
    predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']
    maximum = df[predictors].max()
    
    df[predictors] = df[predictors]/df[predictors].max()

    targets= ['Buy', 'Sell']

    X = df[predictors].values
    y = df[targets].values

    y_buy = to_categorical(y[:,0])

    y_sell = to_categorical(y[:,1])

    count_classes = y_buy.shape[1], y_sell.shape[1]

    print(count_classes)

    # pred_train= buy_model.predict(X_train)
    scores = lr_buy.evaluate(X, y_buy, verbose=0)
    print('Accuracy on buy data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
    
    # pred_test= buy_model.predict(X_test)
    scores2 = lr_sell.evaluate(X, y_sell, verbose=0)
    print('Accuracy on sell data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

    buy_pred= lr_buy.predict(X)
    sell_pred= lr_sell.predict(X)

    buy_predictions = pd.DataFrame(buy_pred, index=labelled_stock_data.index)
    sell_predictions = pd.DataFrame(sell_pred, index=labelled_stock_data.index)

    # LRData[predictors] = LRData[predictors] * maximum

    labelled_stock_data["buy_probability"] = buy_predictions[1]
    labelled_stock_data["sell_probability"] = sell_predictions[1]

    labelled_stock_data.reset_index(level=0, inplace=True)

    labelled_stock_data = labelled_stock_data.drop(['Date'], axis=1)

    print(labelled_stock_data)

    scaler = load(open(dir_path+'/recommender_scaler.pkl', 'rb'))

    data = scaler.transform(labelled_stock_data)

    print(data.shape)

    X = []
    # X_train_buy = []
    Y_buy = []
    # X_train_sell = []
    Y_sell = []

    for i in range(60, data.shape[0]):

        X.append(data[i-60:i])
        # X_train_buy.append(training_data_buy[i-60:i])
        Y_buy.append(data[i, 11])

        # X_train_sell.append(training_data_sell[i-60:i])
        Y_sell.append(data[i, 12])
        # Y_train_predict.append(training_data[i, 3])

    X, Y_buy, Y_sell = np.array(X), np.array(Y_buy), np.array(Y_sell)

    lstm_recommender_buy = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Buy.h5')
    lstm_recommender_sell = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Sell.h5')

    scoresBuy = lstm_recommender_buy.evaluate(X, Y_buy, verbose=0)
    print("Accuracy: %.2f%%" % (scoresBuy[1]*100))

    scoresSell = lstm_recommender_sell.evaluate(X, Y_sell, verbose=0)
    print("Accuracy: %.2f%%" % (scoresSell[1]*100))

def prediction_case_study(stock_code):
    print("The prediction case study")

    # stock = yf.Ticker(stock_code)
    # df = stock.history(period='7y')
    df = yf.download(stock_code, start='2012-01-01', end='2019-12-17')
    # df = yf.download("AAPL", start='2012-01-01', end='2019-12-17')
    df.drop(["Adj Close"], axis = 1, inplace = True) 
    # df.drop(["Dividends", "Stock Splits"], axis = 1, inplace = True) 

    #Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    #Converting the dataframe to a numpy array
    dataset = data.values
    #Get /Compute the number of rows to train the model on
    # training_data_len = math.ceil( len(dataset) *.8) 
    scaler = load(open(dir_path+'/predictor_scaler.pkl', 'rb'))
    scaled_data = scaler.transform(dataset)

    x = []
    y = dataset[60:]
    for i in range(60,len(scaled_data)):
        x.append(scaled_data[i-60:i,0])

    x, y = np.array(x), np.array(y)

    x = np.reshape(x, (x.shape[0],x.shape[1],1))

    model = tf.keras.models.load_model(dir_path+'/lstm_predictor.h5')

    #Getting the models predicted price values
    predictions = model.predict(x)
    predictions = scaler.inverse_transform(predictions)#Undo scaling

    scores = model.evaluate(x, y, verbose=0)

    # print("Accuracy: %.2f%%" % (scores[1]*100))
    print(scores)

    rmse=np.sqrt(np.mean(((predictions - y)**2)))

    print(rmse)


def make_recommendation(stock_code):
    
    lr_buy = tf.keras.models.load_model(dir_path+'/LR_Recommender_Buy.h5')
    lr_sell = tf.keras.models.load_model(dir_path+'/LR_Recommender_Sell.h5')
    lstm_buy = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Buy.h5')
    lstm_sell = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Sell.h5')

    stock = yf.Ticker(stock_code)

    data = stock.history(period='100d')

    data.drop(["Dividends","Stock Splits"], axis = 1, inplace = True) 

    data['Daily_Vol'] = getSdChangeInPrice(data['Close'])
    data['S_10'] = data['Close'].rolling(window=10).mean()
    data['Corr'] = data['Close'].rolling(window=10).corr(data['S_10'])
    data['RSI'] = ta.RSI(np.array(data['Close']), timeperiod=10)
    data['Open-Close'] = data['Open'] - data['Close'].shift(1)
    data['Open-Open'] = data['Open'] - data['Open'].shift(1)

    data = data.dropna()

    classifyData = data.copy()

    predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']
    maximum = classifyData[predictors].max()
    
    classifyData[predictors] = classifyData[predictors]/classifyData[predictors].max()

    classifyValues = classifyData[predictors].values

    buy_stock_preds = lr_buy.predict(classifyValues)
    sell_stock_preds = lr_sell.predict(classifyValues)

    buy_predictions = pd.DataFrame(buy_stock_preds, index=data.index)
    sell_predictions = pd.DataFrame(sell_stock_preds, index=data.index)

    classifyData[predictors] = classifyData[predictors] * maximum

    buy_predictions_temp = buy_predictions[1].copy()
    sell_predictions_temp = sell_predictions[1].copy()
    
    data["Buy"] = buy_predictions_temp.round(decimals=0)
    data["Sell"] = sell_predictions_temp.round(decimals=0)
    data["buy_probability"] = buy_predictions[1]
    data["sell_probability"] = sell_predictions[1]

    print(data)

    data.reset_index(level=0, inplace=True)

    past_60_days = data.tail(61) 

    print(past_60_days)

    scaler = load(open(dir_path+'/recommender_scaler.pkl', 'rb'))

    past_60_days = past_60_days.drop(['Date'], axis=1)
    last_60_days = scaler.transform(past_60_days)

    lstm_probs = []

    print(last_60_days.shape[0])

    for i in range(60, last_60_days.shape[0]):
        print(i)
        lstm_probs.append(last_60_days[i-60:i])

    lstm_probs = np.array(lstm_probs)

    print(lstm_probs.shape)

    buy_prediction = lstm_buy.predict(lstm_probs)
    sell_prediction = lstm_sell.predict(lstm_probs)
  
    lstm_buy_prediction = buy_prediction[0][0]
    lstm_sell_prediction = sell_prediction[0][0]

    lr_buy_prediction = buy_stock_preds[-1][1]
    lr_sell_prediction = sell_stock_preds[-1][1]

    print("Buy lstm", type(lstm_buy_prediction))
    print("Sell lstm", type(lstm_sell_prediction))
    print("Buy lr", type(lr_buy_prediction))
    print("Sell lr", type(lr_sell_prediction))

    action = "Hold"
    likelihood = 0
    if(lstm_buy_prediction < 0.5 or lr_buy_prediction < 0.5 or lstm_sell_prediction < 0.5 or lr_sell_prediction < 0.5):
        if(lstm_buy_prediction > 0.5 and lr_buy_prediction > 0.5):
            action = "Buy"
            likelihood = lstm_buy_prediction
        elif(lstm_sell_prediction > 0.5 and lr_sell_prediction > 0.5):
            action = "Sell"
            likelihood = lstm_sell_prediction

    predictions = make_predictions(stock_code)

    # print(predictions)
    
    return {"predictions": predictions, "action": action, "likelihood": likelihood}

def make_predictions(stock_code):

    model = tf.keras.models.load_model(dir_path+'/lstm_predictor.h5')
    scaler = load(open(dir_path+'/predictor_scaler.pkl', 'rb'))
    stock = yf.Ticker(stock_code)

    apple_quote = stock.history(period='12y')
    # df = yf.download("AAPL", start='2012-01-01', end='2019-12-17')
    apple_quote.drop(["Dividends", "Stock Splits"], axis = 1, inplace = True) 
    # apple_quote = yf.download("MSFT", start='2012-01-01', end='2019-12-17')
    #Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    #Get teh last 60 day closing price 
    last_60_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)

    print(pred_price)

    return pred_price
    
def train_models(stock_code, initial):

    print(stock_code, initial)

    stock = yf.Ticker(stock_code)

    stock_data = stock.history(period='12y')
    stock_data.drop(["Dividends","Stock Splits"], axis = 1, inplace = True) 

    labelled_stock_data = TripleBarrierMethod(stock_data)
    labelled_stock_data = labelled_stock_data.dropna()

    if initial == True:
        print("Lets train the model with this stock code for the first time", "we need to check if there is already a model or not")
        try:
            df = labelled_stock_data.copy()

            target_column = ['Buy'] 
            target_column1 = ['Sell']
            predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']
            maximum = df[predictors].max()
            
            df[predictors] = df[predictors]/df[predictors].max()

            targets= ['Buy', 'Sell']

            X = df[predictors].values
            y = df[targets].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


            print(X_train.shape), print(X_test.shape)
            print(y_train.shape), print(y_test.shape)

            # one hot encode outputs
            y_train = to_categorical(y_train[:,0])
            y_test = to_categorical(y_test[:,0])

            z_train = to_categorical(y_train[:,1])
            z_test = to_categorical(y_test[:,1])

            count_classes = y_test.shape[1], z_test.shape[1]

            print(count_classes)

            buy_model = tf.keras.models.load_model(dir_path+'/LR_Recommender_Buy.h5')
            sell_model = tf.keras.models.load_model(dir_path+'/LR_Recommender_Sell.h5')

            buy_model.fit(X_train, y_train, epochs=100)
            sell_model.fit(X_train, z_train, epochs=100)
            print("Model is here") 

        except:
            print("Model is not there")
            buy_model = make_LRModel(X_train, y_train)
            sell_model = make_LRModel(X_train, z_train)

        # pred_train= buy_model.predict(X_train)
        scores = buy_model.evaluate(X_train, y_train, verbose=0)
        print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
        
        # pred_test= buy_model.predict(X_test)
        scores2 = buy_model.evaluate(X_test, y_test, verbose=0)
        print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))
        
        # pred_train2= sell_model.predict(X_train)
        scores3 = sell_model.evaluate(X_train, z_train, verbose=0)
        print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores3[1], 1 - scores3[1]))   
        
        # pred_test2= sell_model.predict(X_test)
        scores4 = sell_model.evaluate(X_test, z_test, verbose=0)
        print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores4[1], 1 - scores4[1]))

        buy_model.save(dir_path+'/LR_Recommender_Buy.h5')
        sell_model.save(dir_path+'/LR_Recommender_Sell.h5')

        train_LSTMModelRecommendation(labelled_stock_data, initial, buy_model, sell_model)
        train_LSTMModelPrediction(stock_code, initial)

    elif initial == False:
        print("Let's train the model with new data that we just have got")

        lastDaySearch = labelled_stock_data.copy()

        predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']

        maximum = lastDaySearch[predictors].max()
            
        lastDaySearch[predictors] = lastDaySearch[predictors]/lastDaySearch[predictors].max()

        day = lastDaySearch.last('1D')

        x = day[predictors].values
        y = day[['Buy', 'Sell']].values

        # one hot encode outputs
        print("the fuck", y[:,0], y[:,1])
        y_buy = to_categorical(y[:,0], num_classes=2)
        y_sell = to_categorical(y[:,1], num_classes=2)

        print("This is ",y_buy)
        print("This is ",y_sell)

        count_classes = y_buy.shape[1], y_sell.shape[1]

        print(count_classes)

        buy_model = tf.keras.models.load_model(dir_path+'/LR_Recommender_Buy.h5')
        sell_model = tf.keras.models.load_model(dir_path+'/LR_Recommender_Sell.h5')

        buy_model.fit(x, y_buy, epochs=5)
        sell_model.fit(x, y_sell, epochs=5)

        scores = buy_model.evaluate(x, y_buy, verbose=0)
        print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))  

        scores2 = sell_model.evaluate(x, y_sell, verbose=0)
        print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores2[1], 1 - scores2[1]))  



        # buy_model.save(dir_path+'/LR_Recommender_Buy.h5')
        # sell_model.save(dir_path+'/LR_Recommender_Sell.h5')

        train_LSTMModelRecommendation(labelled_stock_data, initial, buy_model, sell_model)
        train_LSTMModelPrediction(stock_code, initial)


    return "Completed training {}".format(stock_code)

def train_LSTMModelRecommendation(data, initial, buy_model = None, sell_model = None):

    if(buy_model is not None):

        print(data)

        LRData = data.copy()

        predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']
        maximum = LRData[predictors].max() 
        
        LRData[predictors] = LRData[predictors]/LRData[predictors].max()

        X = LRData[predictors].values

        buy_pred= buy_model.predict(X)
        sell_pred= sell_model.predict(X)

        buy_predictions = pd.DataFrame(buy_pred, index=data.index)
        sell_predictions = pd.DataFrame(sell_pred, index=data.index)

        LRData[predictors] = LRData[predictors] * maximum

        data["buy_probability"] = buy_predictions[1]
        data["sell_probability"] = sell_predictions[1]

        data.reset_index(level=0, inplace=True)

        if initial == True:
            # data.to_csv(r'data6.csv', index=False)
            # print("Saved data to file")

            train_amount = int(len(data) * 0.8)
            # data_training_buy = data[:train_amount].copy()
            # data_training_sell = data[:train_amount].copy()
            data_training = data[:train_amount].copy()

            data_training = data_training.drop(['Date'], axis=1)
            # data_training_buy = data_training_buy.drop(['Date', 'Sell'], axis=1)
            # data_training_sell = data_training_sell.drop(['Date', 'Buy'], axis=1)

            print(data_training)

            data_testing = data[train_amount:].copy()
            try:
                scaler = load(open(dir_path+'/recommender_scaler.pkl', 'rb'))
                training_data = scaler.transform(data_training)
                # scaler.transform(dataset)
            except:
                scaler = MinMaxScaler()
                # training_data_buy = scaler.fit_transform(data_training_buy)
                training_data = scaler.fit_transform(data_training)
                dump(scaler, dir_path+'/recommender_scaler.pkl') 
            # training_data_sell = scaler.transform(data_training_sell)

            X_train = []
            # X_train_buy = []
            Y_train_buy = []

            # X_train_sell = []
            Y_train_sell = []

            # Y_train_predict = []

            for i in range(60, training_data.shape[0]):

                X_train.append(training_data[i-60:i])
                # X_train_buy.append(training_data_buy[i-60:i])
                Y_train_buy.append(training_data[i, 11])

                # X_train_sell.append(training_data_sell[i-60:i])
                Y_train_sell.append(training_data[i, 12])
                # Y_train_predict.append(training_data[i, 3])

            X_train, Y_train_buy, Y_train_sell = np.array(X_train), np.array(Y_train_buy), np.array(Y_train_sell)
            # X_train_buy, Y_train_buy = np.array(X_train_buy), np.array(Y_train_buy)
            # X_train_sell, Y_train_sell = np.array(X_train_sell), np.array(Y_train_sell)
            # , Y_train_predict , np.array(Y_train_predict)
            # joblib.dump(clf, 'my_dope_model.pkl')

            try:
                lstm_recommender_buy = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Buy.h5')
                lstm_recommender_sell = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Sell.h5')
                lstm_recommender_buy.fit(X_train, Y_train_buy, epochs=10, batch_size=64)
                lstm_recommender_sell.fit(X_train, Y_train_sell, epochs=10, batch_size=64)
            except:
                lstm_recommender_buy = make_RecommendationLSTMModel(X_train, Y_train_buy)
                lstm_recommender_sell = make_RecommendationLSTMModel(X_train, Y_train_sell)
                # predictor = make_LSTMModel(X_train, Y_train_predict)

            # past_60_days_buy = data_training_buy.tail(60)
            # past_60_days_sell = data_training_sell.tail(60)
            past_60_days= data_training.tail(60)

            # df_buy = past_60_days_buy.append(data_testing, ignore_index=True)
            # df_sell = past_60_days_sell.append(data_testing, ignore_index=True)
            # df_sell = df_sell.drop(['Date', 'Buy'], axis=1)
            # df_buy = df_buy.drop(['Date', 'Sell'], axis=1)

            df = past_60_days.append(data_testing, ignore_index=True)
            df = df.drop(['Date'], axis=1)
            # df.head()

            # inputs_buy = scaler.transform(df_buy)
            # inputs_sell = scaler.transform(df_sell)
            inputs = scaler.transform(df)


            X_test = []
            # X_test_buy = []
            Y_test_buy = []

            # X_test_sell = []
            Y_test_sell = []
            # Y_test_predict = []

            for i in range(60, inputs.shape[0]):
                X_test.append(inputs[i-60:i])
                # X_test_buy.append(inputs_buy[i-60:i])
                Y_test_buy.append(inputs[i,11])

                # X_test_sell.append(inputs_sell[i-60:i])
                Y_test_sell.append(inputs[i,12])
                # Y_test_predict.append(inputs[i,3])

            X_test, Y_test_buy,Y_test_sell = np.array(X_test), np.array(Y_test_buy), np.array(Y_test_sell)
            # X_test_buy, Y_test_buy = np.array(X_test_buy), np.array(Y_test_buy)
            # X_test_sell, Y_test_sell = np.array(X_test_sell), np.array(Y_test_sell)
            # , Y_test_predict, np.array(Y_test_predict)
            
            scoresBuy = lstm_recommender_buy.evaluate(X_test, Y_test_buy, verbose=0)
            print("Accuracy: %.2f%%" % (scoresBuy[1]*100))

            scoresSell = lstm_recommender_sell.evaluate(X_test, Y_test_sell, verbose=0)
            print("Accuracy: %.2f%%" % (scoresSell[1]*100))
            
            # scoresPredict = predictor.evaluate(X_test, Y_test_predict, verbose=0)
            # print(scoresPredict)

            lstm_recommender_buy.save(dir_path+'/lstm_Recommender_Buy.h5')
            lstm_recommender_sell.save(dir_path+'/lstm_Recommender_Sell.h5')
            # predictor.save(dir_path+'/stock_predictor.h5')

        
        elif initial == False:
            print("We want to retrain the lstm model here")

            lstm_recommender_buy = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Buy.h5')
            lstm_recommender_sell = tf.keras.models.load_model(dir_path+'/lstm_Recommender_Sell.h5')
            
            final60days = data.tail(61)
            final60days.reset_index(level=0, inplace=False)
            final60days = final60days.drop(['Date'], axis=1)
        
            scaler = load(open(dir_path+'/recommender_scaler.pkl', 'rb'))

            final60days = scaler.transform(final60days)

            # print(final60days)

            X_train = []
            # X_train_buy = []
            Y_train_buy = []

            # X_train_sell = []
            Y_train_sell = []

            # Y_train_predict = []

            for i in range(60, final60days.shape[0]):

                X_train.append(final60days[i-60:i])
                # X_train_buy.append(training_data_buy[i-60:i])
                Y_train_buy.append(final60days[i, 11])

                # X_train_sell.append(training_data_sell[i-60:i])
                Y_train_sell.append(final60days[i, 12])
                # Y_train_predict.append(training_data[i, 3])

            X_train, Y_train_buy, Y_train_sell = np.array(X_train), np.array(Y_train_buy), np.array(Y_train_sell)

            lstm_recommender_buy.fit(X_train, Y_train_buy, epochs=1, batch_size=64)
            lstm_recommender_sell.fit(X_train, Y_train_sell, epochs=1, batch_size=64)

            scoresBuy = lstm_recommender_buy.evaluate(X_train, Y_train_buy, verbose=0)
            print("Accuracy: %.2f%%" % (scoresBuy[1]*100))

            scoresSell = lstm_recommender_sell.evaluate(X_train, Y_train_sell, verbose=0)
            print("Accuracy: %.2f%%" % (scoresSell[1]*100))

            # lstm_recommender_buy.save(dir_path+'/lstm_Recommender_Buy.h5')
            # lstm_recommender_sell.save(dir_path+'/lstm_Recommender_Sell.h5')

def train_LSTMModelPrediction(stock_code, initial):

    if initial == True:
        print("This is initial")
        msft = yf.Ticker(stock_code)
        df = msft.history(period='12y')
        # df = yf.download("AAPL", start='2012-01-01', end='2019-12-17')
        df.drop(["Dividends", "Stock Splits"], axis = 1, inplace = True) 

        #Create a new dataframe with only the 'Close' column
        data = df.filter(['Close'])
        #Converting the dataframe to a numpy array
        dataset = data.values
        #Get /Compute the number of rows to train the model on
        training_data_len = math.ceil( len(dataset) *.8)

        try:
            scaler = load(open(dir_path+'/predictor_scaler.pkl', 'rb'))
            scaled_data = scaler.transform(dataset)
        except:
            scaler = MinMaxScaler(feature_range=(0, 1)) 
            scaled_data = scaler.fit_transform(dataset) 
            dump(scaler, dir_path+'/predictor_scaler.pkl') 

        #Create the scaled training data set 
        train_data = scaled_data[0:training_data_len  , : ]
        #Split the data into x_train and y_train data sets
        x_train=[]
        y_train = []
        for i in range(60,len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])

        #Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        #Reshape the data into the shape accepted by the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        try:
            lstm_predictor = tf.keras.models.load_model(dir_path+'/lstm_predictor.h5')
            lstm_predictor.fit(x_train, y_train, batch_size=1, epochs=1)
        except:
            lstm_predictor = make_PredictionLSTMModel(x_train, y_train)
            lstm_predictor.save(dir_path+'/lstm_predictor.h5')

        #Test data set
        test_data = scaled_data[training_data_len - 60: , : ]
        #Create the x_test and y_test data sets
        x_test = []
        y_test = dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
        for i in range(60,len(test_data)):
            x_test.append(test_data[i-60:i,0])

        #Convert x_test to a numpy array 
        x_test = np.array(x_test)

        #Reshape the data into the shape accepted by the LSTM
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

        #Getting the models predicted price values
        predictions = lstm_predictor.predict(x_test) 
        predictions = scaler.inverse_transform(predictions)#Undo scaling

        #Calculate/Get the value of RMSE
        rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
        print(rmse)

        
    elif initial == False:
        print("This not initial")

        lstm_predictor = tf.keras.models.load_model(dir_path+'/lstm_predictor.h5')

        msft = yf.Ticker(stock_code)
        df = msft.history(period='61d')
        # df = yf.download("AAPL", start='2012-01-01', end='2019-12-17')
        df.drop(["Dividends", "Stock Splits"], axis = 1, inplace = True) 
        # lstm_predictor.save(dir_path+'/lstm_predictor.h5')
        print(df)
        #Create a new dataframe with only the 'Close' column
        data = df.filter(['Close'])
        #Converting the dataframe to a numpy array
        dataset = data.values

        scaler = load(open(dir_path+'/predictor_scaler.pkl', 'rb'))
        scaled_data = scaler.transform(dataset)

        x_train = []
        y_train = []

        for i in range(60,len(scaled_data)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])

        #Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        #Reshape the data into the shape accepted by the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        lstm_predictor.fit(x_train, y_train, batch_size=1, epochs=1)

        # lstm_predictor.save(dir_path+'/lstm_predictor.h5')

        


def make_LRModel(X_train, y_train):

    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=11))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100)

    model.summary()

    return model

def make_RecommendationLSTMModel(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 15)))
    # regressior.add(LSTM(units = 60, activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 6)))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 80, return_sequences = True))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 80, return_sequences= True))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 120))
    model.add(Dropout(0.3))

    model.add(Dense(units = 1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(X_train, Y_train, epochs=10, batch_size=64)

    return model

def make_PredictionLSTMModel(X_train, Y_train):
    # Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(X_train, Y_train, batch_size=1, epochs=1)

    return model
'''

def make_LSTMModel(X_train, Y_train):

    regressior = Sequential()
    regressior.add(LSTM(units = 60, activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 15)))
    # regressior.add(LSTM(units = 60, activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 6)))
    regressior.add(Dropout(0.2))

    regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    regressior.add(Dropout(0.3))

    regressior.add(LSTM(units = 80, activation = 'relu', return_sequences= True))
    regressior.add(Dropout(0.3))

    regressior.add(LSTM(units = 120, activation = 'relu'))
    regressior.add(Dropout(0.3))

    regressior.add(Dense(units = 1))

    regressior.summary()

    regressior.compile(optimizer='adam', loss='mean_squared_error')

    regressior.fit(X_train, Y_train, epochs=10, batch_size=32)

    return regressior
'''

def getSdChangeInPrice(close_prices, lookback = 20):
    daily_vol = close_prices.index.searchsorted(close_prices.index - pd.Timedelta(days=1))
    daily_vol = daily_vol[daily_vol>0]
    daily_vol = pd.Series(data=close_prices.index[daily_vol - 1], index=close_prices.index[close_prices.shape[0]-daily_vol.shape[0]:])

    try:
        daily_vol = (close_prices.loc[daily_vol.index] / close_prices.loc[daily_vol.values].values-1) # daily returns
    except Exception as e:
        print('error: {}\nplease confirm no duplicate indices'.format(type(e).__name__, e))

    daily_vol = daily_vol.ewm(span=lookback).std()

    return daily_vol.iloc[1:]

def TripleBarrierMethod(data):

    upper_lower_multipliers=[2,2] # how risky you want this to be, the lower it is the more riskier (less holding)

    t_final = 10 # it is how many days ahead to check if the price move significantly, size of window
    close_prices = data['Close']
    highs = data['High']
    lows = data['Low']
    daily_vol = getSdChangeInPrice(close_prices) # how much it has changed for each day

    # print ("{}, {}, {}, {}".format(type(close_prices), highs, lows, daily_vol))

    # print(type(daily_vol))
    out = pd.DataFrame(index = daily_vol.index) # creating a dataframe, with dates as index
    out['Open'] = data['Open']
    out['High'] = highs
    out['Low'] = lows
    out['Close'] = close_prices
    out['Volume'] = data['Volume']
    out['Daily_Vol'] = daily_vol
    out['S_10'] = out['Close'].rolling(window=10).mean()
    out['Corr'] = out['Close'].rolling(window=10).corr(out['S_10'])
    out['RSI'] = ta.RSI(np.array(out['Close']), timeperiod=10)
    out['Open-Close'] = out['Open'] - out['Close'].shift(1)
    out['Open-Open'] = out['Open'] - out['Open'].shift(1)

    # out.at["2010-02-08 00:00:00", "Security"] = 1

    # print(out)

    for day, sdChangeInPrice in daily_vol.iteritems():
        # print(day, vol)

        days_passed = len(daily_vol[daily_vol.index[0] : day])
        # print(days_passed)

        if (days_passed + t_final < len(daily_vol.index) and t_final != 0): # checks if enough values for window to be size 10
            vert_barrier = daily_vol.index[days_passed + t_final] # the vertical barrier is today plus window size.
            # if today is the 10th March, then the vertical barrier is the 20th March
        else:
            vert_barrier = np.nan

        # set the top barrier to current price plus how much the price changed in last 20 days
        # it is the value at which if the price reaches in the future, it would be considered a buy today.
        if upper_lower_multipliers[0] > 0:
            top_barrier = close_prices[day] + close_prices[day] * sdChangeInPrice * upper_lower_multipliers[0]  
        else:
            # top_barrier = pd.Series(index=close_prices.index) # NaNs
            top_barrier = np.nan # NaNs

        # set the bottom barrier to current price minuses how much the price changed in last 20 days
        # it is the value at which if the price reaches in the future, it would be a considered a sell today.
        if upper_lower_multipliers[1] > 0:
            bot_barrier = close_prices[day] - close_prices[day] * sdChangeInPrice * upper_lower_multipliers[1] 
            # 450       =  500(price today) -  500(price today) *     0.1         *  1
            # if the price hits 450(bottom barrier) then within the next 10 days(t_final) we consider it a sale today at price 500.
        else:
            # bot_barrier = pd.Series(index=close_prices.index) # NaNs
            bot_barrier = np.nan # NaNs

        # print(top_barrier, bot_barrier)

        breakthrough_date = vert_barrier # if the price doesn't go up or down, then the breakthrough date is set to the vertical barrier, which is the date in 10 days.
            
        # For t_final days after current date (or remaining days in time_frame, whichever ends first)
        # for the next 10 days
        for future_date in daily_vol.index[days_passed : min(days_passed + t_final, len(daily_vol.index))]:
            # print(future_date)
            # if the high of these 10 days is higher than the barrier, then the date this happens becomes our breakthrough date.
            if ((highs[future_date] >= top_barrier or 
                     close_prices[future_date] >= top_barrier and
                     top_barrier != 0)):
                        out.at[day, "Buy"] = 1
                        out.at[day, "Sell"] = 0
                        # out.at[day,"Action"] = 0
                        breakthrough_date = future_date
                        break
            elif (lows[future_date] <= bot_barrier or
                      close_prices[future_date] <= bot_barrier and 
                      bot_barrier != 0):
                    out.at[day, "Buy"] = 0
                    out.at[day, "Sell"] = 1
                    # out.at[day,"Action"] = 2
                    breakthrough_date = future_date
                    break

        # if it not a buy or a sell, here we calculate the value in between
        if (breakthrough_date == vert_barrier):
            # Initial and final prices for Security on timeframe (purchase, breakthrough)
            price_initial = close_prices[day]
            price_final   = close_prices[breakthrough_date]

            if price_final > top_barrier:
                out.at[day, "Buy"] = 1
                out.at[day, "Sell"] = 0
                # out.at[day,"Action"] = 0
            elif price_final < bot_barrier:
                out.at[day, "Buy"] = 0
                out.at[day, "Sell"] = 1
                # out.at[day,"Action"] = 2
            else:
                # out.at[day, "Security"] = max([(price_final - price_initial) / (top_barrier - price_initial),
                #                              (price_final - price_initial) / (price_initial - bot_barrier)], key=abs)
                out.at[day, "Buy"] = 0
                out.at[day, "Sell"] = 0
                # out.at[day,"Action"] = 1
            # print(price_initial, price_final)

    # print(highs["2018-05-04 00:00:00"])
    # print(highs["2018-05-04 00:00:00"])
    return out[:-1]