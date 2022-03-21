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
            ing_list_obj =  IngredientList.objects.last()

            ing_list_obj.get_recipe()
            ing_list_obj.save()

            saved_ingredients_list = IngredientList.objects.all()
            context = {"ing_list_obj": ing_list_obj, "saved_ingredients_list":saved_ingredients_list}
            messages.success(request, ('Your ingredients were successfully added!'))
        else:
            messages.error(request, 'Error saving form')
        return redirect('/output')
    
    ing_form = IngredientListForm()
    saved_ingredients_list = IngredientList.objects.all()
    return render(request=request, template_name='recipe_generator/home.html', context={'ing_form':ing_form, 'saved_ingredients_list':saved_ingredients_list})

def output(request):
    ing_list_obj =  IngredientList.objects.last()# get_object_or_404(IngredientList, pk=ing_list_obj_id)
    return render(request, 'recipe_generator/output.html', {'ing_list_obj': ing_list_obj,})