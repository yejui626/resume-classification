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
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_text_from_pdf(contents):
    pdf_stream = BytesIO(contents)
    reader = PdfReader(pdf_stream)
    text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])
    return text

# LSTM model definition (must match training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))    
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

# Load model and preprocessing objects
word2idx = joblib.load("model/word2idx.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
max_seq_length = joblib.load("model/max_seq_length.pkl")
vocab_size = len(word2idx)
embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=2,
    dropout=0.2
).to(device)
model.load_state_dict(torch.load("model/lstm_classifier.pth", map_location=device))
model.eval()

def text_to_sequence(text, word2idx, max_seq_length):
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    sequence = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(sequence) < max_seq_length:
        sequence.extend([word2idx['<PAD>']] * (max_seq_length - len(sequence)))
    else:
        sequence = sequence[:max_seq_length]
    return sequence

# FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    sequence = text_to_sequence(text, word2idx, max_seq_length)
    sample_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(sample_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    return JSONResponse(content={"predicted_category": label_encoder.inverse_transform([predicted_label])[0]})