from django.db import models
from django.forms import ModelForm
import sys
sys.path.append('.\\scripts')

from interface import get_recipe_with_string_input

# Create your models here.
class TestRecipe(models.Model):
    name = models.CharField(max_length=255)

class IngredientList(models.Model):
    tag = models.CharField(max_length = 50)
    ing_list = models.CharField(max_length = 400)
    complete = models.BooleanField("End of Ingredients?")
    recipe = models.CharField(max_length = 850, default="Recipe_Placeholder")

    def get_recipe(self):
        self.recipe = get_recipe_with_string_input(self.ing_list, self.complete)
        return self.recipe
    
    def __str__(self):
        return self.tag

class IngredientListForm(ModelForm):
    class Meta:
        model = IngredientList
        exclude = ['recipe']
