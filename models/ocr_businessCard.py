import streamlit as st
import cv2
import pytesseract
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

st.title("OCR Business Card Application")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(img, config=config)

        lines = text.strip().split('\n')

        st.write("Card Details:")
        st.write(re.sub(r'[^a-zA-Z ]', '', lines[0]))
        st.write(re.sub(r'[^a-zA-Z ]', '', lines[1]))

        st.image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), caption="Original Image")
    else:
        st.write("Failed to read the image. Please try again.")