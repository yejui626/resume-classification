import streamlit as st
import requests
import PyPDF2

st.title("Resume Category Classifier")
st.write("Upload a resume PDF to predict its category.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if st.button("Predict Category"):
    if uploaded_file is None:
        st.warning("Please upload a PDF file.")
    else:
        # Extract text from uploaded PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])
        
        # Send the text to the FastAPI backend for prediction
        response = requests.post("http://localhost:8000/predict", json={"text": text})
        
        if response.status_code == 200:
            predicted_category = response.json().get("category")
            st.success(f"Predicted Category: {predicted_category}")
        else:
            st.error("Error in prediction. Please try again.")