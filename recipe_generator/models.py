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
        clean = self.recipe.split(' <|endofing|> <|startoftext|> ')

        output_ingredients = clean[0] + '</li>'

        output_ingredients = output_ingredients.replace('<|startofing|>', '<li>')
        output_ingredients = output_ingredients.replace(' <|ingseparator|> ', '</li> <li>')

        #Removing any duplicatation of steps
        first_steps = clean[1][:30]
        if first_steps in clean[1][25:]:
            dup = clean[1][30:].index(first_steps)
            clean[1] = clean[1][:dup+29]
        
        #Removing any existing numbering
        if clean[1][0] == '1':
            output_steps = [item if len(item)>2 else None for item in clean[1].split('. ')]
            while None in output_steps:
                output_steps.remove(None)
            output_steps = '<li>' + '</li> <li>'.join(output_steps)
        else:
            output_steps = '<li>' + clean[1].replace('. ', '</li> <li>')
            
        output_steps = output_steps.replace('<|endoftext|>', '</li>')

        self.output_ingredients = output_ingredients
        self.output_steps = output_steps

    def __str__(self):
        return self.tag

