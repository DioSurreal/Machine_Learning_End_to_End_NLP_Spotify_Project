import os
from flask import Flask, request, jsonify
import joblib
import preprocessing.preprocess as preprocess
import preprocessing.preprocessing_lstm as lstm_preprocess
import torch
from model.model_lstm import SentimentLSTM

# Create an instance of Flask web application
app = Flask(__name__)

# Hyperparameters for LSTM model
vocab_size = 21669 # Vocabulary size, +1 for the 0 padding token
output_size = 1  # Binary classification (POSITIVE or NEGATIVE)
embedding_dim = 400  # Dimension of word embeddings
hidden_dim = 128  # Hidden layer size of the LSTM
n_layers = 2  # Number of LSTM layers

# Load the Logistic Regression model for supervised learning prediction
model_logisitic = joblib.load('model/best_logistic_regression_model.pkl')

# Load the TF-IDF vectorizer used for text feature extraction
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Load the trained LSTM model for sentiment analysis
model_lstm = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model_lstm.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model_lstm.eval()  # Set the model to evaluation mode

# Route for the homepage, returns a simple greeting message
@app.route('/', methods=['GET'])
def __main__():
    return "Hi! eiei I'm Dio"

# Route for logistic regression sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the input text from the POST request
        text = request.json['text']
        
        # Preprocess the input text
        dataprocessed = preprocess.preprocess_text(text)
        
        # Transform the preprocessed text using the TF-IDF vectorizer
        transformed_data = vectorizer.transform([dataprocessed])
        
        # Make prediction using the logistic regression model
        prediction = model_logisitic.predict(transformed_data)
        
        # Map the prediction to a sentiment label (POSITIVE or NEGATIVE)
        result = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
        return result  # Return the prediction result
    except Exception as e:
        # In case of an error, return a JSON response with the error message
        return jsonify({"error": str(e)})

# Route for LSTM-based sentiment prediction
@app.route('/lstm/predict', methods=['POST'])
def lstm_predict():
    try:
        # Retrieve the input text from the POST request
        text = request.json['text']
        
        # Preprocess the input text for LSTM model
        input = lstm_preprocess.preprocess_lstm(text)
        
        # Convert the processed text into a PyTorch tensor
        input = torch.tensor(input, dtype=torch.long)

        # Add a batch dimension for the model input
        input = input.unsqueeze(0)  # Adding batch dimension

        # Initialize hidden state for the LSTM model
        hidden = model_lstm.init_hidden(batch_size=1)  # Hidden state initialization

        # Make prediction using the LSTM model without computing gradients
        with torch.no_grad():
            output, _ = model_lstm(input, hidden)  # Pass input and hidden state to the model
            pred = torch.round(output.squeeze()).item()  # Get the final prediction as 0 or 1
        
        # Map the prediction to a sentiment label (POSITIVE or NEGATIVE)
        result = "POSITIVE" if pred == 1 else "NEGATIVE"
        return result  # Return the prediction result
    except Exception as e:
        # In case of an error, return a JSON response with the error message
        return jsonify({"error": str(e)})

# Run the Flask application when this script is executed
if __name__ == '__main__':
    # Run the app on the specified port (default is 8080) and host (accessible from any IP address)
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
