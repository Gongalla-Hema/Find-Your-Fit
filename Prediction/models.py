from django.db import models
from django.contrib.auth.models import User
from PIL import Image


# class Size(models.Model):
# 	user = models.OneToOneField(User, on_delete=models.CASCADE)
# 	foot_image = models.ImageField(default='foot.jpg',upload_to='Foot_Pics')

# 	def __str__(self):
# 		return f'{self.user.username} Size'

# 	def save(self,*args,**kwargs):
# 		super().save(*args,**kwargs)

# 		img = Image.open(self.foot_image.path)
# 		img.save(self.foot_image.path)

class FootImage(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	image = models.ImageField(default='foot.jpg',upload_to='Foot_Pics')

	def __str__(self):
		return f'{self.user.username} Foot Image'