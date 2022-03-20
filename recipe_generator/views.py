from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
# from django.template import loader

from django.views.generic import TemplateView, ListView
from .models import TestRecipe, IngredientList
from django.db.models import Q

# Create your views here.

class HomePageView(TemplateView):
    template_name = 'recipe_generator/home.html'
    context_object_name = 'saved_ingredient_lists'

    def get_queryset(self):
        return IngredientList.objects.all

class OutputPageView(ListView):
    model = IngredientList
    template_name = 'recipe_generator/output.html'
    context_object_name = 'ing_list_obj'

    # This may not work how you think it does. Use for reference but do not trust
    def get_queryset(self):
        for i in IngredientList.objects.all():
            if i.recipe == "Recipe_Placeholder":
                i.recipe = i.get_recipe()
                i.save()
        return IngredientList.objects

def home(request):
    saved_ingredient_lists = IngredientList.objects.all
    context = {
        'saved_ingredient_lists': saved_ingredient_lists,
    }    
    return render(request, 'recipe_generator/output.html', context)

def output(request, ing_list_obj_id):
    ing_list_obj =  get_object_or_404(IngredientList, pk=ing_list_obj_id)
    return render(request, 'recipe_generator/output.html', {'ing_list_obj': ing_list_obj,})