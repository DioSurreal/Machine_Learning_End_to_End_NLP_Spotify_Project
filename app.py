import os
from flask import Flask, request, jsonify
import joblib
import preprocessing.preprocess as preprocess
import preprocessing.preprocessing_lstm as lstm_preprocess
import torch
from model.model_lstm import SentimentLSTM
app = Flask(__name__)

vocab_size = 21669 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 128
n_layers = 2

#supervised learning  model   
model_logisitic = joblib.load('model/best_logistic_regression_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')


#lstm model
model_lstm = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model_lstm.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model_lstm.eval()

@app.route('/', methods=['GET'])
def __main__():
    return "Hi! eiei I'm Dio"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        dataprocessed = preprocess.preprocess_text(text)
        transformed_data = vectorizer.transform([dataprocessed])
        prediction = model_logisitic.predict(transformed_data)
        result = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
        return result
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/lstm/predict', methods=['POST'])
def lstm_predict():
    try:
        text = request.json['text']
        input = lstm_preprocess.preprocess_lstm(text)
        # input = input.unsqueeze(0)  # เพิ่มมิติ batch
        input = torch.tensor(input, dtype=torch.long)

        # เพิ่มมิติ batch
        input = input.unsqueeze(0)  # เพิ่มมิติ batch
        # สร้าง hidden state ก่อนทำการทำนาย
        hidden = model_lstm.init_hidden(batch_size=1)  # เริ่มต้น hidden state ให้เหมาะสม

        with torch.no_grad():
            output, _ = model_lstm(input, hidden)  # ส่งทั้ง input และ hidden ไปยัง forward
            pred = torch.round(output.squeeze()).item()
        
        result = "POSITIVE" if pred == 1 else "NEGATIVE"
        return result
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)

