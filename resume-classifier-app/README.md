# Resume Classifier Application

This project is a Resume Classifier that utilizes a trained LSTM model to categorize resumes based on their content. The application is split into two main components: a FastAPI backend that handles the model inference and a Streamlit frontend that provides a user interface for uploading resumes and displaying predictions.

## Project Structure

```
resume-classifier-app
├── backend
│   ├── app.py                # Main entry point for the FastAPI application
│   ├── model
│   │   ├── lstm_model.pth    # Trained LSTM model weights
│   │   ├── tokenizer.pkl      # Tokenizer for text preprocessing
│   │   ├── label_encoder.pkl  # Label encoder for converting predicted labels
│   │   └── max_len.pkl        # Maximum length of sequences used during training
│   ├── requirements.txt       # Dependencies for the FastAPI application
│   └── README.md              # Documentation for the backend
├── frontend
│   ├── streamlit_app.py       # Main entry point for the Streamlit application
│   ├── requirements.txt        # Dependencies for the Streamlit application
│   └── README.md               # Documentation for the frontend
└── README.md                   # Overall documentation for the project
```

## Setup Instructions

### Backend

1. Navigate to the `backend` directory.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the FastAPI application:
   ```
   uvicorn app:app --reload
   ```

### Frontend

1. Navigate to the `frontend` directory.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```

## API Usage

The FastAPI backend exposes an endpoint for resume classification. You can send a POST request with the resume text to receive the predicted category.

## Deployment

For free deployment solutions, consider using:

- **Heroku** for the FastAPI backend.
- **Streamlit Sharing** for the Streamlit frontend.

## Acknowledgments

This project utilizes various libraries including FastAPI, Streamlit, PyTorch, and NLTK for natural language processing tasks.