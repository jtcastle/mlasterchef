from django.urls import path
from . import views
from .views import HomePageView, OutputPageView

urlpatterns = [
    path('', views.home, name='home'),
    path('output/', views.output, name='output'),
]