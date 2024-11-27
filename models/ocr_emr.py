# Libraries and dependencies
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import google.generativeai as genai
import cv2
import numpy as np
import pytesseract
import io

# Reading API key from a file
with open('api.txt', 'r') as file:
    GOOGLE_API_KEY = file.read().strip()

# Configuring Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Model configuration settings
MODEL_CONFIG = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

# Safety settings to avoid harmful content
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

# Initializing the generative model with configuration and safety settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings
)

# Function to preprocess the image
def preprocess_image(image):
    gray_image = ImageOps.grayscale(image)  # Convert image to grayscale
    image_np = np.array(gray_image)  # Convert PIL image to numpy array
    blurred_image = cv2.GaussianBlur(image_np, (5, 5), 0)  # Apply Gaussian blur
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Defining a structuring element
    eroded_image = cv2.erode(thresh_image, kernel, iterations=1)  # Eroding the image
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)  # Dilating the image
    processed_image_pil = Image.fromarray(dilated_image)  # Convert numpy array back to PIL image
    return processed_image_pil

# Function to draw bounding boxes around detected text regions
def draw_bounding_boxes(image):
    image_np = np.array(image)  # Convert PIL image to numpy array
    d = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)  # Get OCR data
    draw = ImageDraw.Draw(image)  # Create a drawing context
    n_boxes = len(d['level'])  # Get the number of detected text boxes
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])  # Get bounding box coordinates
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)  # Draw rectangle
    return image

# Function to format the image for the model input
def image_format(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img_bytes.getvalue()
        }
    ]
    return image_parts

# Function to generate output using Google Generative AI
def gemini_output(image, system_prompt, user_prompt):
    image_info = image_format(image)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response.text

system_prompt = """
You are a specialist in comprehending electricity meter readings.
Input images in the form of electricity meter readings will be provided to you,
and your task is to respond to questions based on the content of the input image.
"""

# Streamlit app title
st.title("OCR Electricity Meter Reader Application")

# File uploader for users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Processing the uploaded file if it exists
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing...")

    preprocessed_image = preprocess_image(image)  # Preprocess the image
  
    boxed_image = draw_bounding_boxes(preprocessed_image)  # Draw bounding boxes
    
    st.image(boxed_image, caption='Preprocessed Image with Bounding Boxes.', use_column_width=True)

    user_prompt = "What is the electricity meter reading?"   
    meter_number_output = gemini_output(preprocessed_image, system_prompt, user_prompt)  # Get the meter reading
    st.write("Meter Number: ", meter_number_output)  # Display the meter reading
