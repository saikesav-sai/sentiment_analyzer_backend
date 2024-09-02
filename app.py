from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os


keras_file          =os.path.join(os.path.dirname(__file__), 'model.keras')
tokenizer_file      =os.path.join(os.path.dirname(__file__), 'tokenizer.pkl')
label_encoder_file  =os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

# Load the saved model and tokenizer
rnn_model = load_model(keras_file)
with open(tokenizer_file, 'rb') as f:
    tokenizer = pickle.load(f)
with open(label_encoder_file, 'rb') as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    print(f'Data saved:') 
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
