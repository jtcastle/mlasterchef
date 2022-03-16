from django.urls import path

from .views import HomePageView, OutputPageView

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('output/', OutputPageView.as_view(), name='output'),
]