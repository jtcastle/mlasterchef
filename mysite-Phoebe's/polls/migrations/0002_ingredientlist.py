# Generated by Django 4.0.2 on 2022-02-25 11:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='IngredientList',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tag', models.CharField(max_length=50)),
                ('ing_list', models.CharField(max_length=400)),
                ('complete', models.BooleanField(verbose_name='End of Ingredients?')),
            ],
        ),
    ]
