import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found. Please add your API key to a .env file.")
else:
    # Configure Google Generative AI
    genai.configure(api_key=api_key) 
    model = genai.GenerativeModel('gemini-pro')
    
# Setting web app page name and selecting wide layout(optional)
    st.set_page_config(page_title='Text-To-SQL APP', page_icon=None)

    # Setting column size
    col1, col2 = st.columns((0.3, 1.7))
    
    col1.image('text_to_sql_logo.jpeg')
    col2.markdown("# :rainbow[SQL QUERY AI ASSISTANT APP]")
    st.write("#### :blue[This is SQL Query Generator Web App Using Google Gemini!]")
    
    supportive_info1 = "Based on the prompt text, create a SQL query, and make sure to exclude ''' in the beginning and end."
    supportive_info2 = "Based on the SQL query code, create an example input dataframe before the SQL query code is applied and the output dataframe after the SQL query is applied."
    supportive_info3 = "Explain the SQL query in detail without any example output."
    
    if submit:
        with st.spinner("Generating.."):
            # Generate SQL Query
            response = model.generate_content([supportive_info1, query_input])
            st.write("##### 1. The Generated SQL Query Code :")
            st.code(response.text)
            
            # Generate Example Output
            response2 = model.generate_content([supportive_info2, response.text])
            st.write("##### 2. A Sample Of Expected Output :")
            st.write(response2.text)
            
            # Generate Explanation
            response3 = model.generate_content([supportive_info3, response.text])
            st.write("##### 3. Explanation of the SQL Query code generated :")
            st.write(response3.text)