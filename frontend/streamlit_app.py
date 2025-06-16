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
        # Send file directly to FastAPI backend
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files=files)

        if response.status_code == 200:
            predicted_category = response.json().get("predicted_category")
            st.success(f"Predicted Category: {predicted_category}")
        else:
            st.error("Error in prediction. Please try again.")
