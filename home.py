import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from edge import Canny_detector
from region import Kmeans_cluster
from PIL import ImageEnhance
from PIL import ImageColor
from color_filter import Color_Filter
from Image_enhance import apply_brightness_contrast,hsv_control
from shapes import detect_shape
import requests
from streamlit_lottie import st_lottie
import io
import base64

def load_image(img):
    im = Image.open(img).convert('RGB')
    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    opencvImage = cv2.resize(opencvImage,(600,400))
    return opencvImage

def get_image_download_link(img,filename,text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ocoxrlkm.json")

l2 = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_zyq1qkhh.json")

st.title("Image processing WebApp")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="high", # medium ; high
    #renderer="svg", # canvas
    height=300,
    width=300,
    key=None,
)


uploaded_image = st.file_uploader("Choose a image file", type=["jpg","png"])


if uploaded_image is not None:

    operation  = st.selectbox("Select operation",("Shape Detection","Edge detection","Region cluster","Image Enhancement","Color Filter"))
    img = load_image(uploaded_image)
    #img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    show_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st.image(show_img)
    if operation=="Edge detection":
        strength = st.sidebar.slider("weak",min_value=0.0,max_value=1.0,value=0.0,step=0.002)
        canny = Canny_detector(img,weak_th=strength)
        st.image(canny,clamp=True,channels="GRAY")
        #canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2RGB)
        canny_down = Image.fromarray(canny)
        canny_down = canny_down.convert("RGB")
        link = get_image_download_link(canny_down,"Edge detected","Download the image")
        st.write(link,unsafe_allow_html = True)
    elif operation=="Region cluster":
        n = st.sidebar.slider("Number of region",min_value=2,max_value=10,value=2)
        region_seg = Kmeans_cluster(img,n)
        region_down= Image.fromarray(region_seg)
        link = get_image_download_link(region_down,"Region detected","Download the image")
        st.image(region_seg,channels="RGB") 
        st.write(link,unsafe_allow_html = True)


    elif operation=="Color Filter":
        color1 = st.sidebar.color_picker('lower bound', '#00f900')
        x,y,z = ImageColor.getcolor(color1,"RGB")
        low = np.array([x,y,z])
        color2 = st.sidebar.color_picker('Upper bound', '#00f900')
        a,b,c = ImageColor.getcolor(color2,"RGB")
        high = np.array([a,b,c])
        new_image = Color_Filter(img,low,high)
        new_img_down = Image.fromarray(new_image)
        link = get_image_download_link(new_img_down,"Color detected","Download the image")
        st.image(new_image)
        st.write(link,unsafe_allow_html = True)
    
    elif operation=="Image Enhancement":
        contrast = st.sidebar.slider("Contrast",min_value=-120,max_value=120,value=0,step=10)
        brightness = st.sidebar.slider("Brightness",min_value=-120,max_value=120,value=0,step=10)
        hue = st.sidebar.slider("Hue",min_value=0.1,max_value=2.0,value=1.0,step=0.1)
        saturation = st.sidebar.slider("Saturation",min_value=0.1,max_value=2.5,value=1.0,step=0.1)
        value = st.sidebar.slider("Color Intensity",min_value=0.1,max_value=2.0,value=1.0,step=0.1)
        hsvImg = hsv_control(img,hue,saturation,value)
        new_image=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        new_image = apply_brightness_contrast(new_image,brightness,contrast)
        st.image(new_image,channels="BGR")
        new_img_down = Image.fromarray(new_image)
        link = get_image_download_link(new_img_down,"Enhancement detected","Download the image")
        st.write(link,unsafe_allow_html = True)
    elif operation=="Shape Detection":
        shaped_img,shapes = detect_shape(img)
        st.sidebar.write(shapes)

else:
    st.warning("Please Upload an jpg or png image")

st_lottie(
    l2,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    #renderer="svg", # canvas
    height=400,
    width=400,
    key=None,
)   
