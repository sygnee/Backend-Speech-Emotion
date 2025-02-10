import os
import numpy as np
import librosa
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for frontend

UPLOAD_FOLDER = 'uploads'
RECORDINGS_FOLDER = 'recordings'
PLAYLISTS_FILE = 'playlists.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECORDINGS_FOLDER'] = RECORDINGS_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RECORDINGS_FOLDER):
    os.makedirs(RECORDINGS_FOLDER)

# ✅ Load the pre-trained model
try:
    model = tf.keras.models.load_model('D:/BE/BE project flask api/Speech-Emotion-Recogniton/final_model.keras')
    print(model.summary())  # Debugging: Check if the model is loaded
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

# ✅ Define emotion labels
emotions = ['neutral', 'neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# ✅ In-memory user database (Replace with database in production)
users = {}
playlists = {}

# =========================== SIGNUP API ===========================
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        print("🔹 Received Signup Data:", data)  # Debug

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400

        if username in users:
            return jsonify({"error": "User already exists"}), 400

        users[username] = password
        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        print("❌ Error in signup:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

# =========================== LOGIN API ===========================
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        print("🔹 Received Login Data:", data)  # Debug

        username = data.get("username")
        password = data.get("password")

        if users.get(username) == password:
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print("❌ Error in login:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

# =========================== PLAYLIST MANAGEMENT ===========================
@app.route('/api/playlists', methods=['POST'])
def save_playlist():
    try:
        data = request.json
        username = data.get("username")
        playlist_name = data.get("name")
        songs = data.get("songs", [])

        if not username or not playlist_name:
            return jsonify({"error": "Missing username or playlist name"}), 400

        if username not in playlists:
            playlists[username] = {}

        playlists[username][playlist_name] = songs

        return jsonify({"message": "Playlist saved successfully!"}), 201
    except Exception as e:
        print("❌ Error saving playlist:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/playlists/<username>', methods=['GET'])
def get_playlists(username):
    try:
        user_playlists = playlists.get(username, {})
        return jsonify({"playlists": user_playlists}), 200
    except Exception as e:
        print("❌ Error fetching playlists:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

# =========================== PREDICT EMOTION FROM AUDIO ===========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided!"}), 400

        audio_file = request.files['audio']
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
        audio_file.save(audio_path)

        # ✅ Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded!"}), 500

        # ✅ Process the audio
        processed_audio = process_audio(audio_path)
        print("🔹 Processed Audio Shape:", processed_audio.shape)  # Debug

        prediction = model.predict(processed_audio)
        print("🔹 Model Prediction:", prediction)  # Debug

        predicted_index = np.argmax(prediction)
        if predicted_index < len(emotions):
            predicted_emotion = emotions[predicted_index]
        else:
            return jsonify({"error": "Invalid prediction output!"}), 500

        os.remove(audio_path)
        return jsonify({"emotion": predicted_emotion})
    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

# =========================== AUDIO PROCESSING FUNCTION ===========================
def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
        mel_spectrogram_resized = cv2.resize(mel_spectrogram, (383, 38))
        mel_spectrogram_resized = mel_spectrogram_resized[np.newaxis, ..., np.newaxis]
        return mel_spectrogram_resized
    except Exception as e:
        print("❌ Error processing audio:", str(e))
        return None

# =========================== AUDIO RECORDINGS ===========================
@app.route('/save-audio', methods=['POST'])
def save_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['RECORDINGS_FOLDER'], filename)
        audio_file.save(filepath)

        return jsonify({"message": "Recording saved successfully!"}), 201
    except Exception as e:
        print("❌ Error saving recording:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/recordings', methods=['GET'])
def get_recordings():
    try:
        recordings = os.listdir(app.config['RECORDINGS_FOLDER'])
        return jsonify(recordings), 200
    except Exception as e:
        print("❌ Error fetching recordings:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/recordings/<filename>', methods=['GET'])
def serve_recording(filename):
    return send_from_directory(app.config['RECORDINGS_FOLDER'], filename)

# =========================== RUN FLASK SERVER ===========================
if __name__ == '__main__':
    app.run(debug=True)
