from django.contrib import admin

from .models import TestRecipe

# Register your models here.

class TestRecipeAdmin(admin.ModelAdmin):
    list_display = ("name",)

admin.site.register(TestRecipe, TestRecipeAdmin)