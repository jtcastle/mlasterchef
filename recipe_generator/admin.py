from django.contrib import admin

from .models import TestRecipe, IngredientList

# Register your models here.

class TestRecipeAdmin(admin.ModelAdmin):
    list_display = ("name",)

admin.site.register(TestRecipe, TestRecipeAdmin)
admin.site.register(IngredientList)
