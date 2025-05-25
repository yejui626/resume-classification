# Resume Classifier App - Frontend

This project is a Streamlit application that serves as the frontend for a resume classification system. It allows users to upload PDF resumes and receive predictions on their categories by interacting with a FastAPI backend.

## Project Structure

```
resume-classifier-app
├── backend
│   ├── app.py                # FastAPI application for handling predictions
│   ├── model
│   │   ├── lstm_model.pth    # Trained LSTM model weights
│   │   ├── tokenizer.pkl      # Tokenizer for text preprocessing
│   │   ├── label_encoder.pkl  # Label encoder for converting labels
│   │   └── max_len.pkl        # Maximum length of sequences
│   ├── requirements.txt       # Backend dependencies
│   └── README.md              # Backend documentation
├── frontend
│   ├── streamlit_app.py       # Streamlit application for user interface
│   ├── requirements.txt       # Frontend dependencies
│   └── README.md              # Frontend documentation
├── README.md                  # Overall project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd resume-classifier-app
   ```

2. **Navigate to the frontend directory:**
   ```
   cd frontend
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```
   streamlit run streamlit_app.py
   ```

## Usage

- Upload a PDF resume using the provided file uploader in the Streamlit interface.
- Click on the "Predict Category" button to receive the predicted category of the uploaded resume.

## Deployment

For deployment, consider using the following free solutions:

- **FastAPI Backend:** Deploy the backend using [Heroku](https://www.heroku.com/).
- **Streamlit Frontend:** Deploy the frontend using [Streamlit Sharing](https://streamlit.io/sharing).

Make sure to follow the respective documentation for each platform to set up your deployment correctly.