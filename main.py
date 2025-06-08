from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
from PyPDF2 import PdfReader
from io import BytesIO

# Ensure NLTK resources are available
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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
tokenizer = joblib.load("model/tokenizer.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
max_len = joblib.load("model/max_len.pkl")
vocab_size = 5000
embedding_dim = 128
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("model/lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

# FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    sample_tensor = torch.tensor(padded, dtype=torch.float32)

    with torch.no_grad():
        output = model(sample_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    return JSONResponse(content={"predicted_category": label_encoder.inverse_transform([predicted_label])[0]})

def extract_text_from_pdf(contents):

    pdf_stream = BytesIO(contents)
    reader = PdfReader(pdf_stream)
    text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])
    return text