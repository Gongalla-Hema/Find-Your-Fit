from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import FootImage


@receiver(post_save, sender=User)
def create_footimg(sender, instance, created, **kwargs):
    if created:
        FootImage.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_footimg(sender, instance, **kwargs):
    instance.footimage.save()