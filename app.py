import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import PyPDF2

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Preprocessing function (must match training)
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

# LSTM model definition (must match training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.embedding(x.long())
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        return self.fc(final_out)

# Load model and preprocessing objects
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
max_len = joblib.load("max_len.pkl")
vocab_size = 5000
embedding_dim = 128
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
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
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="post")
        sample_tensor = torch.tensor(padded, dtype=torch.float32)
        with torch.no_grad():
            output = model(sample_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        st.success(f"Predicted Category: {label_encoder.inverse_transform([predicted_label])[0]}")
