from django.db import models
from django.forms import ModelForm
import sys
#This does not work on mac I guess
sys.path.append('.\\scripts')
#This works for macs
sys.path.append('../mlasterchef/scripts/')

from interface import get_recipe_with_string_input

# Create your models here. 
class TestRecipe(models.Model):
    name = models.CharField(max_length=255)

class IngredientList(models.Model):
    tag = models.CharField(max_length = 50)
    ing_list = models.CharField(max_length = 400)
    complete = models.BooleanField("End of Ingredients?")
    recipe = models.TextField(default="Recipe_Placeholder")
    output_ingredients = models.TextField(default="ing_Placeholder")
    output_steps = models.TextField(default="steps_Placeholder")

    def get_recipe(self):
        self.recipe = get_recipe_with_string_input(self.ing_list, self.complete)
        self.clean_recipe()
    
    def clean_recipe(self):
        clean = self.recipe.split(' <|endofing|> <|startoftext|>')
        output_ingredients = clean[0] + '</li>'
        output_steps = '<li>' + clean[1]

        output_ingredients = output_ingredients.replace('<|startofing|>', '<li>')
        output_ingredients = output_ingredients.replace(' <|ingseparator|> ', '</li> <li>')

        output_steps = output_steps.replace('<|endoftext|>', '</li>')
        output_steps = output_steps.replace('. ', '</li> <li>')

        self.output_ingredients = output_ingredients
        self.output_steps = output_steps

    def __str__(self):
        return self.tag

