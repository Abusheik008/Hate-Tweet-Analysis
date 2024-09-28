
# Hate Tweet Analysis Flask Application

## Overview
This project is a web-based application that analyzes tweets to detect hate speech using a fine-tuned RoBERTa model for sequence classification. The app is built using **Flask** as the web framework, **PyTorch** for deep learning inference, and **Transformers** for leveraging the RoBERTa model.

The web app accepts a tweet as input, processes it using a custom `DataProcessor`, tokenizes it using a pre-trained RoBERTa tokenizer, and classifies the tweet as either hate speech or non-hate speech using a fine-tuned RoBERTa model.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Application](#how-to-run-the-application)
- [API Endpoints](#api-endpoints)
- [Logging](#logging)
- [Error Handling](#error-handling)
- [Future Enhancements](#future-enhancements)

## Project Structure
```
Hate-Tweet-Analysis/
│
├── src/
│   └── data_preprocessing.py         # Contains the DataProcessor class for tweet cleaning
│
├── templates/
│   └── index.html                    # Frontend template (if applicable)
│
├── model/
│   └── best.pt                       # Pre-trained RoBERTa model weights
│
├── app.py                            # Main Flask application
├── app.log                           # Log file to track app events and errors
└── requirements.txt                  # Python dependencies
```

## Features
- **Web-based UI**: Accepts tweet input via a simple user interface (index.html).
- **REST API**: Provides an API to predict whether a tweet contains hate speech.
- **Pre-trained RoBERTa Model**: The model is fine-tuned to classify tweets into two categories: hate speech and non-hate speech.
- **Data Preprocessing**: Cleans tweet text by removing unnecessary characters and normalizing input.
- **Logging**: Logs key actions such as input text, predictions, and any errors.
- **Error Handling**: Gracefully handles errors and provides meaningful error messages.

## How It Works
1. **Text Input**: The app receives a tweet via an HTML form or through a POST request to the `/predict` API endpoint.
2. **Text Cleaning**: The raw tweet is processed using a custom `DataProcessor` class to remove unwanted characters (e.g., hashtags, mentions, URLs).
3. **Tokenization**: The cleaned tweet is tokenized using the `RobertaTokenizer`, converting the text into a format that the RoBERTa model can understand.
4. **Prediction**: The tokenized text is fed into the fine-tuned RoBERTa model to classify the tweet. The output is either 0 (non-hate speech) or 1 (hate speech).
5. **Result**: The prediction result is sent back to the user, either as a JSON response or displayed in the UI.

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch
- Transformers library by Hugging Face
- Flask web framework

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/hate-tweet-analysis.git
   cd hate-tweet-analysis
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install all required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or Place the Model**:
   Ensure that the fine-tuned RoBERTa model (`best.pt`) is placed in the `model/` directory.

## How to Run the Application

1. **Start the Flask Server**:
   Run the application using the following command:
   ```bash
   python app.py
   ```

2. **Access the Application**:
   The app will be available locally at `http://127.0.0.1:5000/`. Open this URL in your web browser to access the UI.

3. **Submit a Tweet**:
   Enter the tweet text into the input box and click "Submit". The app will return whether the tweet contains hate speech or not.

### Running with Docker
If you'd like to containerize the app with Docker:
1. Create a `Dockerfile`:
   ```Dockerfile
   FROM python:3.8-slim
   WORKDIR /app
   COPY . /app
   RUN pip install -r requirements.txt
   CMD ["python", "app.py"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t hate-tweet-analysis .
   docker run -p 5000:5000 hate-tweet-analysis
   ```

## API Endpoints

### 1. `/` (Home)
- **Method**: GET
- **Description**: Renders the home page where users can submit a tweet for analysis.

### 2. `/predict` (Predict Hate Speech)
- **Method**: POST
- **Description**: Accepts a tweet and returns whether it contains hate speech.
- **Payload**:
  ```json
  {
    "text": "Your tweet text here"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "The provided tweet contains hate speech."
  }
  ```

## Logging
Logging is enabled in this application, and all logs are saved to the `app.log` file. It captures:
- Access to the home page
- Incoming text for predictions
- Prediction results
- Errors that occur during processing

### Example Logs:
```
2024-09-28 14:20:12,123 - INFO - Home page accessed
2024-09-28 14:21:54,432 - INFO - Received text for prediction: "This is a bad tweet!"
2024-09-28 14:21:55,789 - INFO - Prediction result: The provided tweet contains hate speech.
2024-09-28 14:22:10,234 - ERROR - Error during prediction: 'NoneType' object has no attribute 'get'
```

## Error Handling
If an error occurs during prediction (e.g., invalid input or a model error), the application logs the error and returns a JSON response:
```json
{
  "error": "An error occurred during prediction. Please try again later."
}
```

---
