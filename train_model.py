import os
import PyPDF2
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])
    return text

def load_dataset(base_dir):
    data = {}
    for category in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, category)
        if os.path.isdir(folder_path):
            data[category] = []
            for file in os.listdir(folder_path):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(folder_path, file)
                    text = extract_text_from_pdf(pdf_path)
                    data[category].append(text)
    return data

# 2. Convert data into Pandas dataframe
def get_dataframes():
    train_data = load_dataset("dataset_small/train")
    test_data = load_dataset("dataset_small/test")
    train_df = pd.DataFrame(train_data).melt(var_name="Category", value_name="Resume_Text")
    test_df = pd.DataFrame(test_data).melt(var_name="Category", value_name="Resume_Text")
    return train_df, test_df

# 3. Data Preprocessing
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

# 4. Model definition
class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

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

if __name__ == "__main__":
    train_df, test_df = get_dataframes()
    train_df["Cleaned_Text"] = train_df["Resume_Text"].apply(clean_text)
    test_df["Cleaned_Text"] = test_df["Resume_Text"].apply(clean_text)

    X_train, y_train = train_df["Cleaned_Text"], train_df["Category"]
    X_test, y_test = test_df["Cleaned_Text"], test_df["Category"]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    max_len = max(len(seq) for seq in X_train_seq)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

    train_dataset = ResumeDataset(X_train_pad, y_train_encoded)
    test_dataset = ResumeDataset(X_test_pad, y_test_encoded)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    vocab_size = 5000
    embedding_dim = 128
    hidden_dim = 64
    output_dim = len(label_mapping)
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # Reduce for faster training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save model and preprocessing objects
    torch.save(model.state_dict(), "lstm_model.pth")
    joblib.dump(tokenizer, "tokenizer.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(max_len, "max_len.pkl")
    print("Model and preprocessing objects saved.")
