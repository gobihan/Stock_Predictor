from django.core.management.base import BaseCommand, CommandError
from StockPredictor.views import job,recommendation_case_study, prediction_case_study


class Command(BaseCommand):

    def handle(self, *args, **options):

        recommendation_case_study("AAPL")
        prediction_case_study("AAPL")


