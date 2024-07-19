# 1) Imports and Environment Setup:

from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# 2) Configure Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found. Please set it in the .env file.")

def get_gemini_response(input, image, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([input, image[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def input_image_setup(uploaded_file):
    try:
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()
            image_parts = [
                {
                    "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")
    except Exception as e:
        st.error(f"Error setting up image: {e}")
        return None

# Initialize our Streamlit app
st.set_page_config(page_title="Food Image Calories Counter")

st.header("Food Image Calories Counter")

input_prompt = """
               You are an expert in nutrition and calories estimation.
               You will receive input images of food items &
               you will have to estimate the calories of the food item in the image.
               """

input_text = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image of food...", type=["jpg", "jpeg", "png"])
image = ""   

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")

submit = st.button("Estimate Calories")

if submit:
    if uploaded_file is not None and input_text:
        image_parts = input_image_setup(uploaded_file)
        if image_parts:
            response = get_gemini_response(input_text, image_parts, input_prompt)
            if response:
                st.text_area("Calories Estimation Response:", response)
    else:
        st.warning("Please upload an image and provide an input prompt.")
