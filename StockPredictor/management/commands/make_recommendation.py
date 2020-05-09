from django.core.management.base import BaseCommand, CommandError
from StockPredictor.views import job,make_recommendation
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

        stocks_colection = db.collection(u'stocks')
        docs = stocks_colection.stream()

        for doc in docs:
            recommendation = make_recommendation(doc.id)

            amount = recommendation['likelihood']
            action = recommendation['action']
            prediction = float(recommendation['predictions'][0][0])

            test = db.collection(u'recommendations').add({
                'action': action,
                'amount':amount,
                'date': firestore.SERVER_TIMESTAMP
            })
            

            stocks_ref = stocks_colection.document(doc.id)

            print(stocks_ref)

            stocks_ref.update({
                u'recommendations':firestore.ArrayUnion([test[1]])
            })

            db.collection(u'predictions').document(doc.id).set({
                'value': prediction
            })
