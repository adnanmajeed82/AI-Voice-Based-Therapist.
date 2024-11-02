from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import pipeline
from gtts import gTTS
import os
import speech_recognition as sr

app = Flask(__name__)
# Using a conversational model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")
recognizer = sr.Recognizer()

# Directory to store audio responses
AUDIO_FOLDER = 'static/audio_responses'
os.makedirs(AUDIO_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

def recognize_speech():
    """Convert user's voice input to text."""
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Sorry, I'm having trouble with the speech service."

def get_response(user_text):
    """Generate a conversational response from the chatbot model."""
    response = chatbot(user_text, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

def text_to_speech(text, filename="response.mp3"):
    """Convert text response to audio and save as mp3 file."""
    tts = gTTS(text=text, lang='en')
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    tts.save(audio_path)
    return filename

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    """Handle conversation flow: convert speech to text, generate response, convert to audio."""
    user_text = request.json.get("user_text")
    if not user_text:
        user_text = recognize_speech()  # Get voice input if no text provided
    response_text = get_response(user_text)  # Generate chatbot response
    audio_file = text_to_speech(response_text)  # Convert response to speech
    return jsonify({"response_text": response_text, "audio_file": audio_file})

@app.route('/static/audio_responses/<filename>')
def serve_audio(filename):
    """Serve the audio response file."""
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
