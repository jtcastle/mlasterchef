from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
# from django.template import loader

from django.views.generic import TemplateView, ListView
from .models import TestRecipe, IngredientList
from django.db.models import Q

from .forms import IngredientListForm
from django.contrib import messages

# Create your views here.

class HomePageView(TemplateView):
    template_name = 'recipe_generator/home.html'
    context_object_name = 'ing_form'

    def get_queryset(self):
        return IngredientList.objects.all

class OutputPageView(ListView):
    model = IngredientList
    template_name = 'recipe_generator/output.html'
    context_object_name = 'ing_list_obj'

    def get_queryset(self):
        query = self.request.GET.get('q')
        return query

#    def get_queryset(self):
#        for i in IngredientList.objects.all():
#            if i.recipe == "Recipe_Placeholder":
#                i.recipe = i.get_recipe()
#                i.save()
#        return IngredientList.objects

def home(request):
    if request.method == "POST":
        ing_form = IngredientListForm(request.POST)
        if ing_form.is_valid():
            ing_form.save()
            messages.success(request, ('Your movie was successfully added!'))
        else:
            messages.error(request, 'Error saving form')
        return redirect("main/home.html")
    ing_form = IngredientListForm()
    saved_ingredient_lists = IngredientList.objects.all()
    return render(request=request, template_name="main/home.html", context={'ing_form':ing_form, 'saved_ingredient_lists':saved_ingredient_lists})

def output(request, ing_list_obj_id):
    ing_list_obj =  get_object_or_404(IngredientList, pk=ing_list_obj_id)
    return render(request, 'recipe_generator/output.html', {'ing_list_obj': ing_list_obj,})