from django.urls import path
from . import views
from .views import HomePageView, OutputPageView

urlpatterns = [
    path('', views.home, name='home'),
    path('loading', views.loading, name='loading'),
    path('<int:ing_list_obj_id>/', views.recipe, name='recipe'),
    path('output/', views.output, name='output'),
]