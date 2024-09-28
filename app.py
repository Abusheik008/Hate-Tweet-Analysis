from flask import Flask, request, jsonify, render_template
import torch, os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from src.data_preprocessing import DataProcessor
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logger
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



# Load pre-trained model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), "src/results/best.pt"))
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model.eval()  # Set model to evaluation mode

# Instantiate DataProcessor to use clean_tweet method
processor = DataProcessor(train_file='data/train.csv', test_file= 'data/test.csv')

@app.route('/')
def home():
    """
    Render the home page.

    Returns:
        HTML content for the homepage.
    """
    logging.info('Home page accessed')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether the given text is a hate tweet or not.

    This function receives text input via a POST request, cleans the text using DataProcessor's
    clean_tweet function, processes it with the tokenizer, passes it to the pre-trained model, 
    and returns a professional prediction result.

    Returns:
        JSON response containing the prediction result.
    """
    try:
        data = request.json
        raw_text = data.get('text', '')

        # Log incoming text
        logging.info(f'Received text for prediction: {raw_text}')

        # Clean the input text using clean_tweet function
        clean_text = processor.clean_tweet(raw_text)
        logging.info(f'Cleaned text: {clean_text}')

        # Tokenize the cleaned text
        inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)

        # Perform prediction using the model
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        # Map prediction result to a more professional, user-friendly message
        if prediction == 0:
            result = "The provided tweet does not contain hate speech."
        else:
            result = "The provided tweet contains hate speech."

        logging.info(f'Prediction result: {result}')
        return jsonify({'prediction': result})

    except Exception as e:
        # Log error details
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction. Please try again later.'}), 500

if __name__ == '__main__':
    # Log application startup
    logging.info('Starting the Flask application...')
    app.run(debug=True)
