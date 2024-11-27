# import streamlit as st
# import numpy as np
# from PIL import Image
# from models.ocr_model import OCRModel
# from image_processing import preprocess_image

# st.title("OCR App")

# uploaded_file = st.file_uploader("Select an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     image = np.array(image)

#     preprocessed_image = preprocess_image(image)

#     ocr_model = OCRModel()

#     text, accuracy = ocr_model.recognize_text(preprocessed_image)

#     st.write("Recognized Text:", text)
#     st.write("Accuracy:", accuracy)
