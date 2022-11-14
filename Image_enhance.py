import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def hsv_control(img,h,s,v):

    hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsvImg[...,0] = hsvImg[...,0]*h
    hsvImg[...,1] = hsvImg[...,1]*s
    hsvImg[...,2] = hsvImg[...,2]*v
    return hsvImg


