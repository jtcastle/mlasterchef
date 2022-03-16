from django.shortcuts import render
from django.http import HttpResponse
# from django.template import loader

from django.views.generic import TemplateView, ListView
# from .models import Products
# from django.db.models import Q

# Create your views here.

class HomePageView(TemplateView):
    template_name = 'recipe_generator/home.html'