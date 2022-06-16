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

#class OutputPageView(ListView):
#    model = IngredientList
#    template_name = 'recipe_generator/output.html'
#    context_object_name = 'ing_list_obj'
#
#    def get_queryset(self):
#        query = self.request.GET.get('q')
#        return query

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
            ing_form = ing_form.save()
            saved_ingredients_list = IngredientList.objects.all
            messages.success(request, ('Your ingredients were successfully added!'))
        else:
            messages.error(request, 'Error saving form')
        latest_id = IngredientList.objects.last().id
        return redirect('loading', latest_id)
    
    ing_form = IngredientListForm()
    saved_ingredients_list = IngredientList.objects.all()
    return render(request=request, template_name='recipe_generator/home.html', context={'ing_form':ing_form, 'saved_ingredients_list':saved_ingredients_list})

def output(request, ing_list_obj_id):
    ing_list_obj = get_object_or_404(IngredientList, pk=ing_list_obj_id)
    ing_list_obj.get_recipe()
    ing_list_obj.save()
    return redirect('recipe', ing_list_obj.id)

def recipe(request, ing_list_obj_id):
    ing_list_obj =  get_object_or_404(IngredientList, pk=ing_list_obj_id)
    if (request.GET.get('refreshbtn')):
        return redirect('loading', ing_list_obj.id)
        #ing_list_obj.get_recipe()
        #ing_list_obj.save()
        #return redirect('recipe', ing_list_obj.id)
    
    if (request.GET.get('complete_btn')):
        ing_list_obj.complete = not ing_list_obj.complete
        ing_list_obj.save()
        return redirect('recipe', ing_list_obj.id)

    return render(request, 'recipe_generator/output.html', {'ing_list_obj': ing_list_obj,})


def loading(request, ing_list_obj_id):
    return render(request, 'recipe_generator/loading.html', context={'ing_list_obj_id': ing_list_obj_id})

class SearchResultsView(ListView):
    model = IngredientList
    template_name = 'recipe_generator/search_results.html'

    def get_queryset(self):
        query = self.request.GET.get('q')
        object_list = IngredientList.objects.filter(
            Q(ing_list__icontains=query) | Q(tag__icontains=query)
        )
        return object_list
