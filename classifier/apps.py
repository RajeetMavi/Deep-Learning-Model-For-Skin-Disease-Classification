"""
Django app configuration for the classifier app.
"""

from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'
    verbose_name = 'Skin Lesion Classifier'
