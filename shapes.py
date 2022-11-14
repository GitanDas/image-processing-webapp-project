import cv2
import numpy as np
import streamlit as st
from PIL import Image,ImageColor
import io
import base64

def load_image(img):
    im = Image.open(img).convert('RGB')
    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    opencvImage = cv2.resize(opencvImage,(600,400))
    return opencvImage
    
def get_image_download_link(img,filename,text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def detect_shape(f1):
    img = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
    #st.image(img)
    _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    font = cv2.FONT_HERSHEY_COMPLEX
    detcetd_shaped = {}
    detcetd_shaped["Triangle"]=0
    detcetd_shaped["Circle"] = 0
    detcetd_shaped["Pentagon"] =0
    detcetd_shaped["Rectangle"]=0
    detcetd_shaped["ellipse"]=0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), font, 1, (0))
            detcetd_shaped["Triangle"] = detcetd_shaped["Triangle"]+1
        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
            detcetd_shaped["Rectangle"] = detcetd_shaped["Rectangle"]+1
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
            detcetd_shaped["Pentagon"] = detcetd_shaped["Pentagon"]+1
        elif 6 < len(approx) < 15:
            cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
            detcetd_shaped["ellipse"] = detcetd_shaped["ellipse"]+1
        else:
            cv2.putText(img, "Circle", (x, y), font, 0.5, (0))
            detcetd_shaped["Circle"] = detcetd_shaped["Circle"]+1
    #st.write("Original Image(Black and White)")
    #st.image(threshold)
    #st.write("Detected Shape Image")
    st.image(img)
    img_test = Image.fromarray(img)
    link = get_image_download_link(img_test,"Shapes","Download here")
    st.write(link,unsafe_allow_html=True)
    return img,detcetd_shaped
    
    
