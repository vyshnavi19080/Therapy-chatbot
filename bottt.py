import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import random
import re
import ssl
import warnings
import nltk
import pandas as pd
import speech_recognition as sr
import pygame
from gtts import gTTS
from playsound import playsound
import tkinter as tk
from tkinter import messagebox, simpledialog

# Check if NLTK data is already downloaded to avoid warnings
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Suppress NLTK warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize the speech recognizer
recognizer = sr.Recognizer()

emotion_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear"}

# Create GUI window
window = tk.Tk()
window.title("Therapy Chatbot")
window.geometry("800x600")

# Heading
heading = tk.Label(window, text="Welcome to the Therapy Chatbot!", font=("Helvetica", 20, "bold"))
heading.pack(pady=20)


def text_input():
    user_input = simpledialog.askstring(
        "Text Input", "Enter your feelings:", parent= window, initialvalue='\n' * 2)
    return user_input

def speech_input():
    with sr.Microphone() as source:
        print("Speak out your feelings...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            return user_input
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

def speak_text(text):
    tts = gTTS(text)
    tts.save("C:/Users/Durga Prasad/Downloads/response.mp3")

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load("C:/Users/Durga Prasad/Downloads/response.mp3")

    # Play the audio
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Quit pygame mixer
    pygame.mixer.quit()

def therapy_chatbot():
    opposite_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 1}
    negation_list = ["not"]
    data = pd.read_csv("C:/Users/Durga Prasad/Downloads/response_sheet (1).csv").values
    result = {}
    for key, value in data:
        if key not in result:
            result[key] = [value]
        else:
            result[key].append(value)

    data = pd.read_csv("C:/Users/Durga Prasad/Downloads/training.csv")
    description_list = []

    # Data processing
    for description in data.text:
        # Filter words
        description = re.sub("[^a-zA-Z]", " ", description)
        # Change to lowercase
        description = description.lower()

        # Tokenize words
        description = nltk.word_tokenize(description)

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        description = (lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"), pos="v"), pos="a")
                       for word in description)

        description = " ".join(description)
        description_list.append(description)

    x = description_list
    y = data.label.values

    # Create a feature extractor
    vectorizer = CountVectorizer()

    # Convert text data into numerical feature vectors
    X = vectorizer.fit_transform(x)

    # Train the Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, y)


    # Button functions
    def text_input_callback():
        user_input = text_input()
        process_input(user_input)

    def speech_input_callback():
        user_input = speech_input()
        process_input(user_input)

    def exit_callback():
        window.destroy()

    def process_input(user_input):
        if not user_input:
            messagebox.showinfo("Error", "Sorry, I couldn't understand your input. Please try again.")
            return

        user_input = re.sub("[^a-zA-Z]", " ", user_input)
        user_input = user_input.lower()

        user_input = nltk.word_tokenize(user_input)

        user_input = (lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"), pos="v"), pos="a")
                      for word in user_input)

        user_input = " ".join(user_input)
        user_input_vector = vectorizer.transform([user_input])
        y_pred = classifier.predict(user_input_vector)[0]
        split_user_input = user_input.split(" ")
        # Checking for negation
        for word in split_user_input:
            if word in negation_list:
                y_pred = opposite_map[y_pred]
                break
        response_list = result[y_pred]
        response = random.choice(response_list)
        # Print user input and chatbot response
        #print("User: " + user_input)
        print("Chatbot: " + response)
        speak_text(response)
        
    # Create buttons with custom colors and size
    button_bg_color = "#4CAF50"  # Green color
    button_fg_color = "black"    # White text color
    button_width = 15
    button_height = 2

    # Buttons
    text_input_button = tk.Button(window, text="Text Input", command=text_input_callback)
    text_input_button.pack(pady=25)
    text_input_button.configure(bg="green", fg="black", width=15, height=2, font=("Arial", 15, "bold"))

    speech_input_button = tk.Button(window, text="Speech Input", command=speech_input_callback)
    speech_input_button.pack(pady=25)
    speech_input_button.configure(bg="green", fg="black", width=15, height=2, font=("Arial", 15, "bold"))
    
    exit_button = tk.Button(window, text="Exit", command=exit_callback)
    exit_button.pack(pady=25)
    exit_button.configure(bg="green", fg="black", width=15, height=2, font=("Arial", 15, "bold"))

    # Start the GUI event loop
    window.mainloop()

therapy_chatbot()
