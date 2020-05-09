from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path("predict_stock", views.predict_stock, name='predict_stock'),
    # path("make_recommendation", views.make_recommendation, name="make_recommendation"),
    path('add_stock/<str:stock_code>', views.add_stock, name="add_stock"),
    path('find_stock/<str:stock_code>', views.find_stock, name="find_stock")
]