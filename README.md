# Stock_Predictor

This is the GitHub link - https://github.com/gobihan/Stock_Predictor.git

This is the web application where StockMatic's machine learning models are maintained and hosted.

In views.py, are all the methods that are used to train the machine learning models and to make recommendations and predicitions for the application to display. There is an API for the iOS application to check whether a stock is supported by the yahoo finance API, so can get the historical data for it.

In the management folder, is where the commands are defined to call the methods for the training and another for getting recommendations and predictions from the model. These can be run using python manage.py make_recommendation and python manage.py train_model, after changing directory to Stock_Predictor. The heroku scheduler, runs these commands at a specific time of the day, so that it can make recommendations daily and train the models daily so that they are upto date with recent data.

The web application is deployed to https://stock-predictorr.herokuapp.com/. The API to check whether yahoo finance supports https://stock-predictorr.herokuapp.com/predictor/find_stock/msft. 

To run the application locally, a conda environment needs to be made. It can be made using the command conda create -n myenv python=3.7.

Then packages defined in requirements.txt, need to be installed into the environment, which can be done with this command,pip install -r requirements.txt.

Then need to run the command cd Stock_Predictor, then run python manage.py runserver. This then allows you to run the code locally.

The iOS application 

This is the link to the GitHub repo for the iOS application- https://github.com/gobihan/Stock.git.

This is a repository for the iOS application, StockMatic.

The file was too big to add to qmplus, so will be required to clone the repo to use it.

This application is used by the user to view the recommendations and predictions by the machine learning model, transactions that they would have made and view and update their profile information.

The main storyboard displays all the views in the application. Each view has a swift file that is a view controller, which holds the code that can manipulate the application logic using the objects on the view including the text fields and buttons. There are also swift files for the data from the database, these are the models, this is what is the controller is manipulating and displaying on the views.

To run this application locally, an Apple device with XCode installed is required, since it is the only IDE that can be used to develop iOS appliations. This should folder should be opened in XCode, and to run the application select a  target simulator and press the play button, which will create a simulator with the application downloaded and running.

To run the application to on an Apple device, a ligthning cable should be used to connect the device to the machine with XCode installed on it. The device has to be trusted to run the application, which is done in the settings for the Apple device, in the Apple developer settings.


