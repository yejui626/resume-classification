# README for Backend of Resume Classifier App

## Overview
This is the backend component of the Resume Classifier App, which utilizes FastAPI to serve predictions for resume categorization. The backend handles API requests, loads the trained LSTM model, and processes incoming data for classification.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd resume-classifier-app/backend
   ```

2. **Install Dependencies**
   Ensure you have Python 3.7 or higher installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Application**
   You can start the FastAPI server using Uvicorn:
   ```bash
   uvicorn app:app --reload
   ```
   The application will be available at `http://127.0.0.1:8000`.

## API Usage

### Endpoint
- **POST /predict**
  - **Description**: Classifies the uploaded resume PDF and returns the predicted category.
  - **Request Body**: 
    - `file`: The PDF file of the resume to be classified.
  - **Response**: 
    - Returns a JSON object containing the predicted category.

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path_to_resume.pdf"
```

### Example Response
```json
{
  "predicted_category": "Software Engineer"
}
```

## Model and Preprocessing
The backend loads the following files from the `model` directory:
- `lstm_model.pth`: Contains the trained model weights.
- `tokenizer.pkl`: Used for text preprocessing.
- `label_encoder.pkl`: Converts predicted labels back to their original form.
- `max_len.pkl`: Maximum length of sequences used during training.

## Deployment
For free deployment, consider using:
- **Heroku**: For deploying the FastAPI backend.
- **Streamlit Sharing**: For deploying the Streamlit frontend.

## License
This project is licensed under the MIT License. See the LICENSE file for details.