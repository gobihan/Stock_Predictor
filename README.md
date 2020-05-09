# Stock_Predictor

This is the web application where StockMatic's machine learning models are maintained and hosted.

In views.py, are all the methods that are used to train the machine learning models and to make recommendations and predicitions for the application to display. There is an API for the iOS application to check whether a stock is supported by the yahoo finance API, so can get the historical data for it.

In the management folder, is where the commands are defined to call the methods for the training and another for getting recommendations and predictions from the model. These can be run using python manage.py make_recommendation and python manage.py train_model, after changing directory to Stock_Predictor. The heroku scheduler, runs these commands at a specific time of the day, so that it can make recommendations daily and train the models daily so that they are upto date with recent data.

The web application is deployed to https://stock-predictorr.herokuapp.com/. The API to check whether yahoo finance supports https://stock-predictorr.herokuapp.com/predictor/find_stock/msft. 

To run the application locally, a conda environment needs to be made. It can be made using the command conda create -n myenv python=3.7.

Then packages defined in requirements.txt, need to be installed into the environment, which can be done with this command,pip install -r requirements.txt.

Then need to run the command cd Stock_Predictor, then run python manage.py runserver. This then allows you to run the code locally.


