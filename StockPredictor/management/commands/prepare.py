from django.core.management.base import BaseCommand, CommandError
from StockPredictor.views import job,train_models


class Command(BaseCommand):

    def handle(self, *args, **options):

        train_models("aapl", True)


