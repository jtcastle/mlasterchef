from django.urls import path
from . import views
from .views import HomePageView, SearchResultsView

urlpatterns = [
    path('', views.home, name='home'),
    path('loading/<int:ing_list_obj_id>', views.loading, name='loading'),
    path('<int:ing_list_obj_id>/', views.recipe, name='recipe'),
    path('output/<int:ing_list_obj_id>', views.output, name='output'),
    path('search/', SearchResultsView.as_view(), name='search_results'),
]