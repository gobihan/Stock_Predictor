from django.core.management.base import BaseCommand, CommandError
from StockPredictor.views import job,make_recommendation


class Command(BaseCommand):

    def handle(self, *args, **options):

        recommend = make_recommendation("msft")

        print(recommend)


