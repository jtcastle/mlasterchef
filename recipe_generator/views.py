from django.shortcuts import render
from django.http import HttpResponse
# from django.template import loader

from django.views.generic import TemplateView, ListView
from .models import TestRecipe
from django.db.models import Q

# Create your views here.

class HomePageView(TemplateView):
    template_name = 'recipe_generator/home.html'

class OutputPageView(ListView):
    model = TestRecipe
    template_name = 'recipe_generator/output.html'

    # This may not work how you think it does. Use for reference but do not trust
    def get_queryset(self):
        query = self.request.GET.get('q')
        return query

