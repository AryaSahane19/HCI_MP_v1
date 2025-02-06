from pathlib import Path
import eel
import pyttsx3
import speech_recognition as sr
from datetime import datetime, date
import time
import webbrowser
from pynput.keyboard import Key, Controller
import pyautogui
import sys
import os
from os import listdir
from os.path import isfile, join
import wikipedia
import app
from threading import Thread
from typing import Optional

# -------------Object Initialization---------------
r = sr.Recognizer()
keyboard = Controller()
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# Set preferred voice (Windows)
if os.name == 'nt':
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

# ----------------Variables------------------------
file_exp_status = False
files = []
path = str(Path.home())  # Start from user home directory
is_awake = True  # Bot status

# ------------------Functions----------------------
def reply(audio: str) -> None:
    """Convert text to speech and update GUI"""
    app.ChatBot.addAppMsg(audio)
    print("Proton:", audio)
    engine.say(audio)
    engine.runAndWait()

def wish() -> None:
    """Greet user based on time of day"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        reply("Good Morning!")
    elif 12 <= hour < 18:
        reply("Good Afternoon!")   
    else:
        reply("Good Evening!")  
    reply("I am Proton, how may I help you?")

def record_audio() -> Optional[str]:
    """Convert speech to text using Google's API"""
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            return r.recognize_google(audio).lower()
        except sr.WaitTimeoutError:
            print("Listening timed out")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            reply(f"Speech recognition error: {e}")
        return None

def getUserInput(voice_data):
    """Process user input from different sources"""
    if 'proton' in voice_data:
        respond(voice_data)

def respond(voice_data: str) -> None:
    """Process user commands"""
    global file_exp_status, files, is_awake, path
    
    if not voice_data:
        return

    print("Processing:", voice_data)
    app.eel.addUserMsg(voice_data)
    voice_data = voice_data.replace('proton', '').strip()

    if not is_awake:
        if 'wake up' in voice_data:
            is_awake = True
            wish()
        return

    # Command processing
    if 'hello' in voice_data:
        wish()
    elif 'your name' in voice_data:
        reply('My name is Proton!')
    elif 'date' in voice_data:
        reply(date.today().strftime("%B %d, %Y"))
    elif 'time' in voice_data:
        reply(datetime.now().strftime("%I:%M %p"))
    elif 'search' in voice_data:
        query = voice_data.split('search', 1)[1].strip()
        reply(f'Searching for {query}')
        webbrowser.open(f'https://google.com/search?q={query}')
    elif 'location' in voice_data:
        reply('Which place would you like me to locate?')
        if (location := record_audio()):
            webbrowser.open(f'https://google.com/maps/place/{location}')
    elif any(cmd in voice_data for cmd in ('bye', 'exit', 'terminate')):
        reply("Goodbye! Have a great day.")
        is_awake = False
        app.ChatBot.close()
        sys.exit()
    elif 'copy' in voice_data:
        pyautogui.hotkey('ctrl', 'c')
        reply('Copied to clipboard')
    elif 'paste' in voice_data:
        pyautogui.hotkey('ctrl', 'v')
        reply('Pasted from clipboard')
    else:
        reply("I'm not programmed for that task yet!")

# New methods for web interface
#@eel.expose
def triggerVoiceInput():
    """Trigger voice input from web interface"""
    voice_data = record_audio()
    if voice_data:
        getUserInput(voice_data)

#@eel.expose
def getUserInput(msg):
    """Process user input from web interface"""
    if msg:
        respond(msg)

# ------------------Driver Code--------------------
if __name__ == "__main__":
    gui_thread = Thread(target=app.ChatBot.start, daemon=True)
    gui_thread.start()

    # Wait for GUI to initialize
    while not app.ChatBot.started:
        time.sleep(0.1)

    wish()
    
    while True:
        try:
            voice_data = None
            if app.ChatBot.isUserInput():
                voice_data = app.ChatBot.popUserInput()
            else:
                voice_data = record_audio()

            if voice_data and 'proton' in voice_data:
                respond(voice_data)

        except KeyboardInterrupt:
            reply("Shutting down...")
            app.ChatBot.close()
            sys.exit()
        except Exception as e:
            print("Error:", e)
            time.sleep(1)