from django.shortcuts import render

# Create your views here.
from django.http import HttpResponseRedirect, HttpResponse, Http404
from django.shortcuts import render, get_object_or_404
from django.views import generic
from django.urls import reverse
from django.utils import timezone

from .models import Question, Choice, IngredientList

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'saved_ingredient_lists'

    def get_queryset(self):
        return IngredientList.objects.all

class DetailView(generic.DetailView):
    model = IngredientList
    template_name = 'polls/detail.html'
    context_object_name = 'ing_list_obj'

    def get_queryset(self):
        for i in IngredientList.objects.all():
            if i.recipe == "Recipe_Placeholder":
                i.recipe = i.get_recipe()
                i.save()
        return IngredientList.objects

class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/result.html'

def index(request):
    saved_ingredient_lists = IngredientList.objects.all
    context = {
        'saved_ingredient_lists': saved_ingredient_lists,
    }    
    return render(request, 'polls/index.html', context)#HttpResponse(template.render(context, request))

# Leave the rest of the views (detail, results, vote) unchanged
def detail(request, ing_list_obj_id):
    ing_list_obj =  get_object_or_404(IngredientList, pk=ing_list_obj_id)
    return render(request, 'polls/detail.html', {'ing_list_obj': ing_list_obj,})

def results(request, ing_list_obj_id):
    ing_list_obj = get_object_or_404(IngredientList, pk=ing_list_obj_id)
    return render(request, 'polls/results.html', {'ing_list_obj': ing_list_obj})

def vote(request, ing_list_obj_id):
    ing_list_obj = get_object_or_404(IngredientList, pk=ing_list_obj_id)
    return render(request, 'polls/results.html', {'ing_list_obj': ing_list_obj})
    '''
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
    '''
    