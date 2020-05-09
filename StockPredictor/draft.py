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

dir_path = os.path.dirname(os.path.realpath(__file__))
# scaler = MinMaxScaler()

def job():
    print("I'm working...")

def index(request):
    '''
    schedule.every(1).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
    '''
    return HttpResponse("Hello, world. You're at the polls index.")

def find_stock(request, stock_code):

    # try:
    ticker = yf.Ticker(stock_code)

    print(ticker.info)


    return JsonResponse({"name":ticker.info['shortName'], "sector": ticker.info['sector'], "stock_code": str.upper(stock_code)})
        
    # except:
    #     return HttpResponse("There is no stock with the stock code " + stock_code)

    # print(ticker.info)
    

def getSdChangeInPrice(close_prices, lookback = 20):
    daily_vol = close_prices.index.searchsorted(close_prices.index - pd.Timedelta(days=1))
    daily_vol = daily_vol[daily_vol>0]
    daily_vol = pd.Series(data=close_prices.index[daily_vol - 1], index=close_prices.index[close_prices.shape[0]-daily_vol.shape[0]:])

    try:
        daily_vol = (close_prices.loc[daily_vol.index] / close_prices.loc[daily_vol.values].values-1) # daily returns
    except Exception as e:
        print('error: {}\nplease confirm no duplicate indices'.format(type(e).__name__, e))

    daily_vol = daily_vol.ewm(span=lookback).std()

    # print(daily_vol)

    return daily_vol.iloc[1:]

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

    print(classifyValues)

    buy_stock_preds = lr_buy.predict(classifyValues)
    sell_stock_preds = lr_sell.predict(classifyValues)

    buy_predictions = pd.DataFrame(buy_stock_preds, index=data.index)
    sell_predictions = pd.DataFrame(sell_stock_preds, index=data.index)

    classifyData[predictors] = classifyData[predictors] * maximum

    data["buy_probability"] = buy_predictions[1]
    data["sell_probability"] = sell_predictions[1]

    data.reset_index(level=0, inplace=True)

    past_60_days = data.tail(61) 

    print(past_60_days)

    scaler = load(open(dir_path+'/scaler.pkl', 'rb'))

    past_60_days = past_60_days.drop(['Date'], axis=1)

    print(past_60_days)

    last_60_days = scaler.transform(past_60_days)

    print(last_60_days)

    lstm_probs = []

    print(last_60_days.shape[0])

    for i in range(60, last_60_days.shape[0]):
        print("Hello")

        lstm_probs.append(last_60_days[i-60:i])

    lstm_probs = np.array(lstm_probs)

    print(lstm_probs.shape)

    lstm_buy.summary()

    buy_prediction = lstm_buy.predict(lstm_probs)
    sell_prediction = lstm_sell.predict(lstm_probs)

    print("Buy", buy_prediction)
    print("Sell" ,sell_prediction)

    

    # data = data.dropna()

    # data.reset_index(level=0, inplace=True)

    # print(data)    
    '''
    futurePredictions = predict_stock_prices(data)

    last_day = data.tail(1)

    print("type", type(last_day))

    print("Date" , last_day.iloc[0]['Date'].to_pydatetime().strftime("%Y-%m-%d"))
    print("Buy" , last_day.iloc[0]["buy_probability"])
    print("Sell" , last_day.iloc[0]["sell_probability"])
    
    buy = last_day.iloc[0]["buy_probability"]
    sell = last_day.iloc[0]["sell_probability"]
    date = last_day.iloc[0]['Date'].to_pydatetime()

    # .strftime("%Y-%m-%d")

    predictions_list = []

    print(type(futurePredictions))

    for x in np.nditer(futurePredictions):
        print(type(x.item(0)))

        date = date + datetime.timedelta(days=1)

        # prediction = {"date":date.strftime("%Y-%m-%d"), "stock_value": x.item(0)} #remember to check if the date is a saturday, then need to add two more days, so shows monday. Forget about bank holidays for now, unnecessary

        # predictions_list.append(prediction)
        predictions_list.append(x.item(0))
    # return HttpResponse("We are going to make a recommendation on whether this is a buy or sell")
    '''
    return {"predictions": [], "buy": buy, "sell": sell}

    

    '''
    target_column = ['Buy'] 
    target_column1 = ['Sell']
    predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']
    maximum = df[predictors].max()
    
    df[predictors] = df[predictors]/df[predictors].max()
    # print(df[predictors])

    targets= ['Buy', 'Sell']

    X = df[predictors].values
    y = df[targets].values

    # data = data.drop('Date', axis=1)
    # data = data.drop('Buy', axis=1)
    # target_column = ['Sell'] 
    predictors = list(data.columns)
    data[predictors] = data[predictors]/data[predictors].max()

    x = data.values

    print(x.shape)

    predictions_buy = model_buy.predict(x)
    predictions_sell = model_sell.predict(x)

    buy_probability = []
    sell_probability = [] 

    for i in range(len(x)):
        print("Predicted to buy",np.argmax(predictions_buy[i]), "Probabilty for buy", predictions_buy[i][1])
        print("Predicted to sell",np.argmax(predictions_sell[i]), "Probabilty for sell", predictions_sell[i][1])

        buy_probability.append(predictions_buy[i][1])
        # sell_probability = buy_probability.append(predictions_sell[i][1])

    data['Probability'] = buy_probability

    print(data)

    inputs = scaler.fit_transform(data)

    X_test = []
    Y_test = []

    print(inputs.shape[0])

    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])


    # inputs = scaler.transform(df)

    X_test = np.array(X_test)

    print(X_test.shape)

    regressior = tf.keras.models.load_model(dir_path+'/Stock_Predictor.h5')

    Y_pred = regressior.predict(X_test)

    scale = 1/scaler.scale_[3]
    # 1.35832654e-02
    scale

    print(scale)

    print(Y_pred)

    Y_pred = Y_pred * scale

    print(Y_pred)

    '''
def predict_stock_prices(data):

    print("Let's predict the stock prices")

    # print(data)

    predictor = tf.keras.models.load_model(dir_path+'/Stock_Predictor.h5')

    data.reset_index(level=0, inplace=True)

    past_60_days = data.tail(60) 

    df = past_60_days
    df = df.drop(['Date'], axis=1)

    print(df)

    X_test = []
    Y_test = []

    scaler = load(open(dir_path+'/scaler.pkl', 'rb'))

    inputs = scaler.transform(df)

    print(inputs.shape[0])

    X_test.append(inputs[0:60])

    X_test = np.array(X_test)

    Y_pred = predictor.predict(X_test)

    scale = 1/scaler.scale_[3]

    print(scale)

    Y_pred = Y_pred * scale
    # Y_test = Y_test * scale

    return Y_pred
    '''

    '''
def add_stock(request, stock_code):
    train_models(stock_code, True)
    return JsonResponse({'message':"Successfully added stock"})

    
def train_models(stock_code, initial):

    print(stock_code, initial)

    stock = yf.Ticker(stock_code)

    stock_data = stock.history(period='12y')
    stock_data.drop(["Dividends","Stock Splits"], axis = 1, inplace = True) 
    # print(stock_data)

    labelled_stock_data = TripleBarrierMethod(stock_data)
    labelled_stock_data = labelled_stock_data.dropna()

    if initial == True:
        print("Lets train the model with this stock code for the first time", "we need to check if there is already a model or not")
        try:
            # labelled_stock_data.to_csv(r'data.csv')
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

        train_LSTMModel(labelled_stock_data, initial, buy_model, sell_model)

    elif initial == False:
        print("Let's train the model with new data that we just have got")

        lastDaySearch = labelled_stock_data.copy()

        predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Vol', 'S_10', 'Corr', 'RSI', 'Open-Close', 'Open-Open']

        maximum = lastDaySearch[predictors].max()
            
        lastDaySearch[predictors] = lastDaySearch[predictors]/lastDaySearch[predictors].max()

        day = lastDaySearch.last('1D')

        # print(day)

        x = day[predictors].values
        y = day[['Buy', 'Sell']].values

        print(x)
        print(y)

        print(x.shape)
        print(y.shape)

        # one hot encode outputs
        print("the fuck", y[:,0], y[:,1])
        y_buy = to_categorical(y[:,0], num_classes=2)
        y_sell = to_categorical(y[:,1], num_classes=2)

        count_classes = y_buy.shape[1], y_sell.shape[1]

        print(count_classes)

        buy_model = tf.keras.models.load_model(dir_path+'/LR_Recommender_Buy.h5')
        sell_model = tf.keras.models.load_model(dir_path+'/LR_Recommender_Sell.h5')

        buy_model.fit(x, y_buy, epochs=1)
        sell_model.fit(x, y_sell, epochs=1)

        buy_model.save(dir_path+'/LR_Recommender_Buy.h5')
        sell_model.save(dir_path+'/LR_Recommender_Sell.h5')

        train_LSTMModel(labelled_stock_data, initial, buy_model, sell_model)


    return "Completed training {}".format(stock_code)

def train_LSTMModel(data, initial, buy_model = None, sell_model = None):

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
            data_training = data[:train_amount].copy()

            buy_labels = data_training['Buy']
            sell_labels = data_training['Sell']

            data_training = data_training.drop(['Date', 'Sell', 'Buy'], axis=1)
            # data_training_sell = data_training_sell.drop(['Date', 'Buy'], axis=1)

            data_testing = data[train_amount:].copy()

            

            scaler = MinMaxScaler()
            training_data = scaler.fit_transform(data_training)
            dump(scaler, 'scaler.pkl') 
            # training_data_sell = scaler.transform(data_training_sell)

            X_train = []
            Y_train_buy = []

            # X_train = []
            Y_train_sell = []

            print(type(training_data))

            for i in range(60, training_data.shape[0]):
                X_train.append(training_data[i-60:i])
                Y_train_buy.append(buy_labels[i])

                # X_train_sell.append(training_data_sell[i-60:i])
                Y_train_sell.append(sell_labels[i])

            X_train, Y_train_buy, Y_test_sell = np.array(X_train), np.array(Y_train_buy), np.array(Y_train_sell)

            # joblib.dump(clf, 'my_dope_model.pkl')

            try:
                # This needs to be fixedddd
                lstm_recommender_buy = tf.keras.models.load_model(dir_path+'/Stock_Predictor.h5')
                lstm_recommender_buy.fit(X_train, Y_train_buy, epochs=10, batch_size=64)
            except:
                lstm_recommender_buy = make_RecommendationLSTMModel(X_train, Y_train_buy)
                lstm_recommender_sell = make_RecommendationLSTMModel(X_train, Y_train_sell)

            past_60_days = data_training.tail(60)
            # past_60_days_sell = data_training_sell.tail(60)

            df = past_60_days.append(data_testing, ignore_index=True)
            test_buy_labels = df['Buy']
            test_sell_labels = df['Sell']
            df = df.drop(['Date', 'Buy', 'Sell'], axis=1)
            # df_buy = df_buy.drop(['Date', 'Sell'], axis=1)
            # df.head()

            inputs_buy = scaler.transform(df)
            # inputs_sell = scaler.transform(df_sell)

            X_test = []
            Y_test_buy = []

            # X_test_sell = []
            Y_test_sell = []

            for i in range(60, inputs_buy.shape[0]):

                X_test.append(inputs_buy[i-60:i])
                Y_test_buy.append(test_buy_labels[i])

                # X_test_sell.append(inputs_sell[i-60:i])
                Y_test_sell.append(test_sell_labels[i])

            X_test, Y_test_buy, Y_test_sell = np.array(X_test), np.array(Y_test_buy), np.array(Y_test_sell)

            scoresBuy = lstm_recommender_buy.evaluate(X_test, Y_test_buy, verbose=0)
            print("Accuracy: %.2f%%" % (scoresBuy[1]*100))

            scoresSell = lstm_recommender_sell.evaluate(X_test, Y_test_sell, verbose=0)
            print("Accuracy: %.2f%%" % (scoresSell[1]*100))

            lstm_recommender_buy.save(dir_path+'/lstm_Recommender_Buy.h5')
            lstm_recommender_sell.save(dir_path+'/lstm_Recommender_Sell.h5')


            '''
        
        elif initial == False:
            print("We want to retrain the lstm model here")

            final70days = data.tail(61)
            final70days.reset_index(level=0, inplace=False)
            final70days = final70days.drop(['Date'], axis=1)

            print(final70days)

            scaler2 = load(open(dir_path+'/scaler.pkl', 'rb'))

            # final70days = scaler2.transform(final70days)

            # print(final70days)

            # final = np.array(final70days)

            # x = []
            # y = []

            # x[].append(final[:60])
            # y[].append(final[60:])

            X_test = []
            Y_test = []

            inputs = scaler2.transform(final70days)

            print(inputs.shape[0])

            for i in range(60, inputs.shape[0]):
                X_test.append(inputs[i-60:i])
                Y_test.append(inputs[i, 3])

            X_test, Y_test = np.array(X_test), np.array(Y_test)

            predictor = tf.keras.models.load_model(dir_path+'/Stock_Predictor.h5')

            predictor.fit(X_test, Y_test, epochs=10, batch_size=32)

            predictor.save(dir_path+'/Stock_Predictor.h5')
            '''

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
    model.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 13)))
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

def make_LSTMModel(X_train, Y_train):

    regressior = Sequential()
    regressior.add(LSTM(units = 60, activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 13)))
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