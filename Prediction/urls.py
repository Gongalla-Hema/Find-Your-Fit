from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name="Pred-Home"),
    path('about/',views.about, name="Pred-About"),
    path('size/',views.size, name="Pred-Size"),
    path('recommendation/',views.recomend, name="Pred-Recomend")
]
