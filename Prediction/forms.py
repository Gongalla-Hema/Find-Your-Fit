from django import forms
from django.db import models
from django.contrib.auth.models import User
from .models import FootImage

class footimage(forms.ModelForm):
	class Meta:
		model = FootImage
		fields = ['image']

