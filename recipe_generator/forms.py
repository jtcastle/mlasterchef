from .models import IngredientList
from django import forms


class IngredientListForm(forms.ModelForm):
    class Meta:
        model = IngredientList
        fields = ('tag', 'ing_list', 'complete',)
