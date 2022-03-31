from .models import IngredientList
from django import forms

## Below code was the original functioning form
# class IngredientListForm(forms.ModelForm):
#     class Meta:
#         model = IngredientList
#         fields = ('tag', 'ing_list', 'complete',)

TRUE_FALSE_CHOICES = (
    (True, 'Yes'),
    (False, 'No')
)

class IngredientListForm(forms.ModelForm):
    class Meta:
        model = IngredientList
        fields = ('tag', 'ing_list', 'complete',)
        labels = {
            "tag": "",
            "ing_list": "",
            "complete": "Would like the generator to limit the recipe to the ingredients listed"
        }
    
    # I dont really know what this does but it works and allows me to add placeholders
    def __init__(self, *args, **kwargs):
        super(IngredientListForm, self).__init__(*args, **kwargs)
        
        self.fields['complete'].widget = forms.Select(choices=TRUE_FALSE_CHOICES)
        self.fields['tag'].widget.attrs['placeholder'] = 'Please enter a name for your recipe'
        self.fields['ing_list'].widget.attrs['placeholder'] = 'Please enter a comma separated list of ingredients'