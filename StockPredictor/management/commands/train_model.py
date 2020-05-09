from django.core.management.base import BaseCommand, CommandError
from StockPredictor.views import job, train_models
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import date
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

pathToAccountKey = dir_path+'/stockmatic-481f1-firebase-adminsdk-8t26v-12d57862b2.json'

cred = credentials.Certificate(pathToAccountKey)

# Use the application default credentials
# cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)

db = firestore.client()


class Command(BaseCommand):

    def handle(self, *args, **options):


        print("this is a test")
        stocks_colection = db.collection(u'stocks')
        docs = stocks_colection.stream()

        for doc in docs:
            print(u'{} => {}'.format(doc.id, doc.to_dict()))
            train_models(doc.id, False)


        # train_models("AAPL", True)
