from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from .models import FootImage
from .forms import footimage
from django.contrib import messages


from skimage.io import imread
from scipy import ndimage
from imutils import contours
import argparse
import imutils
import cv2
from sklearn.cluster import KMeans
import random as rng
import os
import numpy as np


# Create your views here.
def home(request):
	return render(request,'Prediction/index.html')

def about(request):
	return render(request,'Prediction/about.html')

def recomend(request):
  return render(request,'Prediction/recommendation.html')


def preprocess(path):
    img = imread('media/Foot_Pics/'+str(path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = img/255
    return img

def cropOrig(bRect, oimg):
    x,y,w,h = bRect
    pcropedImg = oimg[y:y+h,x:x+w]
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    crop1 = pcropedImg[y1+y2:h1-y2,x1+x2:w1-x2]
    ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]
    croppedImg = oimg[iy:iy+ih,ix:ix+iw]
    return croppedImg, pcropedImg

def overlayImage(croppedImg, pcropedImg):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)
    new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg
    return new_image

def kMeans_cluster(img):
    # For clustering the image using k-means, we first need to convert it into a 2-dimensional 
    # array
    # (H*W, N) N is channel = 3
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    # Reshape back the image from 2D to 3D image
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D*255)
    return clusteredImg

def getBoundingBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])    
    return boundRect, contours, contours_poly, img

def drawCnt(bRect, contours, cntPoly, img):
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)   
    paperbb = bRect
    for i in range(len(contours)):
      color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
      cv2.drawContours(drawing, cntPoly, i, color)
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), \
              (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)
    return drawing

def edgeDetection(clusteredImage):
  edged1 = cv2.Canny(clusteredImage, 0, 255)
  edged = cv2.dilate(edged1, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
  return edged


def calcFeetSize(pcropedImg, fboundRect):
  x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
  y2 = int(h1/10)
  x2 = int(w1/10)
  fh = y2 + fboundRect[2][3]
  fw = x2 + fboundRect[2][2]
  ph = pcropedImg.shape[0]
  pw = pcropedImg.shape[1]
  opw = 210
  oph = 297
  ofs = 0.0
  if fw>fh:
    ofs = (oph/pw)*fw
  else :
    ofs = (oph/ph)*fh
  return ofs

@login_required
def size(request):
    if request.method == 'POST':
        p_form = footimage(request.POST,request.FILES,instance=request.user.footimage)
        if p_form.is_valid():
            p_form.save()
            messages.success(request, 'Here are your results')
            a =  p_form.cleaned_data.get("image")
            oimg = p_form
            preprocessedOimg = preprocess(a)
            clusteredImg = kMeans_cluster(preprocessedOimg)
            edgedImg = edgeDetection(clusteredImg)
            boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
            pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
            croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
            newImg = overlayImage(croppedImg, pcropedImg)
            fedged = edgeDetection(newImg)
            fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
            fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
            output = calcFeetSize(pcropedImg, fboundRect)/10

            output_m = output
            output_w = output

            if output_m < 23.5:
              UK_M = "Invaild Image"
              US_M = "Invaild Image"
              Euro_M = "Invaild Image"
            elif output_m >= 23.5 and output_m < 24.1:
              UK_M = 5.5
              US_M = 6
              Euro_M = 39
            elif output_m >= 24.1 and output_m < 24.4:
              UK_M = 6
              US_M = 6.5
              Euro_M = 39
            elif output_m >= 24.4 and output_m < 24.8:
              UK_M = 6.5
              US_M = 7
              Euro_M = 40
            elif output_m >= 24.8 and output_m < 25.4:
              UK_M = 7
              US_M = 7.5
              Euro_M = "40 - 41"
            elif output_m >= 25.4 and output_m < 25.7:
              UK_M = 7.5
              US_M = 6
              Euro_M = 41
            elif output_m >= 25.7 and output_m < 26:
              UK_M = 8
              US_M = 8.5
              Euro_M = "41 - 42"
            elif output_m >= 26 and output_m < 26.7:
              UK_M = 8.5
              US_M = 9
              Euro_M = 42
            elif output_m >= 26.7 and output_m < 27:
              UK_M = 9
              US_M = 9.5
              Euro_M = "42 - 43"
            elif output_m >= 27 and output_m < 27.3:
              UK_M = 9.5
              US_M = 10
              Euro_M = 43
            elif output_m >= 27.3 and output_m < 27.9:
              UK_M = 10
              US_M = 10.5
              Euro_M = "43 - 44"
            elif output_m >= 27.9 and output_m < 28.3:
              UK_M = 10.5
              US_M = 11
              Euro_M = 44
            elif output_m >= 28.3 and output_m < 28.6:
              UK_M = 11
              US_M = 11.5
              Euro_M = "44 - 45"
            elif output_m >= 28.6 and output_m < 29.4:
              UK_M = 11.5
              US_M = 12
              Euro_M = 45
            elif output_m >= 29.4 and output_m < 30.2:
              UK_M = 12.5
              US_M = 13
              Euro_M = 46
            elif output_m >= 30.2 and output_m < 31:
              UK_M = 13.5
              US_M = 14
              Euro_M = 47
            elif output_m >= 31 and output_m < 31.8:
              UK_M = 14.5
              US_M = 15
              Euro_M = 48
            elif output_m >= 31.8:
              UK_M = 15.5
              US_M = 16
              Euro_M = 49

            print("UK Size: ",UK_M)
            print("US Size: ",US_M)
            print("Euro Size: ",Euro_M)

            if output_w < 20.8:
              UK_W = "Invalid Image"
              US_W = "Invalid Image"
              Euro_W = "Invalid Image"
            elif output_w >= 20.8 and output_w < 21.3:
              UK_W = 2
              US_W = 4
              Euro_W = 35
            elif output_w >= 21.3 and output_w < 21.6:
              UK_W = 2.5
              US_W = 4.5
              Euro_W = 35
            elif output_w >= 21.6 and output_w < 22.2:
              UK_W = 3
              US_W = 5
              Euro_W = "35 - 36"
            elif output_w >= 22.2 and output_w < 22.5:
              UK_W = 3.5
              US_W = 5.5
              Euro_W = 36
            elif output_w >= 22.5 and output_w < 23:
              UK_W = 4
              US_W = 6
              Euro_W = "36 - 37"
            elif output_w >= 23 and output_w < 23.5:
              UK_W = 4.5
              US_W = 6.5
              Euro_W = 37
            elif output_w >= 23.5 and output_w < 23.8:
              UK_W = 5
              US_W = 7
              Euro_W = "37 - 38"
            elif output_w >= 23.8 and output_w < 24.1:
              UK_W = 5.5
              US_W = 7.5
              Euro_W = 38
            elif output_w >= 24.1 and output_w < 24.6:
              UK_W = 6
              US_W = 8
              Euro_W = "38 - 39"
            elif output_w >= 24.6 and output_w < 25.1:
              UK_W = 6.5
              US_W = 8.5
              Euro_W = 39
            elif output_w >= 25.1 and output_w < 25.4:
              UK_W = 7
              US_W = 9
              Euro_W = "39 - 40"
            elif output_w >= 25.4 and output_w < 25.9:
              UK_W = 7.5
              US_W = 9.5
              Euro_W = 40
            elif output_w >= 25.9 and output_w < 26.2:
              UK_W = 8
              US_W = 10
              Euro_W = "40 - 41"
            elif output_w >= 26.2 and output_w < 26.7:
              UK_W = 8.5
              US_W = 10.5
              Euro_W = 41
            elif output_w >= 26.7 and output_w < 27.1:
              UK_W = 9
              US_W = 11
              Euro_W = "41 - 42"
            elif output_w >= 27.1 and output_w < 27.6:
              UK_W = 9.5
              US_W = 11.5
              Euro_W = 42
            elif output_w >= 27.6:
              UK_W = 10
              US_W = 12
              Euro_W = "42 - 43"

            print("US Size: ",US_W)
            print("UK Size: ",UK_W)
            print("Euro Size: ",Euro_W)

            output_m = round(output_m, 2)
            output_w = round(output_w, 2)

            sizes = [
            {
                'Gender' : "Male",
                'UK_Size': UK_M,
                'US_Size': US_M,
                'Euro_Size': Euro_M,
                'Centimeters' : output_m
            },
            {
                'Gender' : "Female",
                'UK_Size': UK_W,
                'US_Size': US_W,
                'Euro_Size': Euro_W,
                'Centimeters': output_w
            }
            ]
            context = {'sizes':sizes}
            return render(request,'Prediction/output.html',context)
    else:
        p_form = footimage(instance=request.user.profile)

    context = {
        'p_form': p_form
    }

    return render(request, 'Prediction/size.html', context)


