"""
Django URL patterns for the classifier app.
"""

from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/', views.analyze, name='analyze'),
    path('results/', views.results, name='results'),
]
