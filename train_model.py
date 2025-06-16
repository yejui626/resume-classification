import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import joblib

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_seq_length=500):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {}
        self.idx2word = {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    def tokenize_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_tokens
    def build_vocabulary(self, texts):
        all_tokens = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_lemmatize(cleaned_text)
            all_tokens.extend(tokens)
        word_counts = Counter(all_tokens)
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    def text_to_sequence(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_lemmatize(cleaned_text)
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        if len(sequence) < self.max_seq_length:
            sequence.extend([self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence)))
        else:
            sequence = sequence[:self.max_seq_length]
        return sequence

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = self.preprocessor.text_to_sequence(text)
        return torch.tensor(sequence), torch.tensor(label, dtype=torch.long)

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

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

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

def get_dataframes():
    train_data = load_dataset("dataset_small/train")
    test_data = load_dataset("dataset_small/test")
    train_df = pd.DataFrame(train_data).melt(var_name="Category", value_name="Resume_Text")
    test_df = pd.DataFrame(test_data).melt(var_name="Category", value_name="Resume_Text")
    return train_df, test_df

def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=50):
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=5)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.best_model)
            break
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return all_preds, all_targets

def main():
    df = pd.read_csv('pdf_to_resume.csv')
    preprocessor = TextPreprocessor()
    preprocessor.build_vocabulary(df['resume'].values)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['occupation'])
    X = df['resume'].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    train_dataset = ResumeDataset(X_train, y_train, preprocessor)
    val_dataset = ResumeDataset(X_val, y_val, preprocessor)
    test_dataset = ResumeDataset(X_test, y_test, preprocessor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=100,
        hidden_dim=128,
        output_dim=len(label_encoder.classes_),
        n_layers=2,
        dropout=0.2
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device
    )
    predictions, targets = evaluate_model(model, test_loader, device)
    accuracy = np.mean(np.array(predictions) == np.array(targets))
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=label_encoder.classes_))
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'model/lstm_classifier.pth')
    print('Model saved as lstm_classifier.pth')
    # Export word2idx, label_encoder, and max_seq_length
    joblib.dump(preprocessor.word2idx, 'model/word2idx.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    joblib.dump(preprocessor.max_seq_length, 'model/max_seq_length.pkl')
    print('word2idx, label_encoder, and max_seq_length exported to model directory.')

if __name__ == "__main__":
    main()
