from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the saved model and tokenizer
rnn_model = load_model(r'model.keras')
with open(r'tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open(r'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "Welcome to the RNN Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']

    # Preprocess the input review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=300)

    # Make prediction
    prediction = rnn_model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)
    class_label = label_encoder.inverse_transform(predicted_label)

    return jsonify({'predicted_label': class_label[0]})

if __name__ == '__main__':
    app.run()
