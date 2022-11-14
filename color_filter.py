import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageColor
import streamlit as st


def Color_Filter(image,low,high):
     
     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
     r,g,b = cv2.split(image)

     m,n = r.shape

     r_low = low[0]
     r_high = high[0]

     g_low = low[1]
     g_high = high[1]

     b_low = low[2]
     b_high = high[2]

     for i in  range(m):

          for j in range(n):

               if r[i][j]>=r_low and r[i][j]<=r_high:
                    continue
               else:
                    r[i][j]=255

     for i in  range(m):

          for j in range(n):

               if g[i][j]>=g_low and g[i][j]<=g_high:
                    continue
               else:
                    g[i][j]=255

     for i in  range(m):

          for j in range(n):

               if b[i][j]>=b_low and b[i][j]<=b_high:
                    continue
               else:
                    b[i][j]=255

     result_image = cv2.merge((r,g,b))
     return result_image


