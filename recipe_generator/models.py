from django.db import models

# Create your models here.
class TestRecipe(models.Model):
    name = models.CharField(max_length=255)
