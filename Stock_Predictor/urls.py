from django.contrib import admin
from django.urls import include,path

urlpatterns = [
    path('predictor/', include('StockPredictor.urls')),
    path('admin/', admin.site.urls),
]
