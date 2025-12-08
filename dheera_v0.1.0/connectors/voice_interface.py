#!/usr/bin/env python3
"""
Dheera Voice Interface
Speech-to-text and text-to-speech capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class VoiceInterface:
    """Voice input/output for Dheera."""
    
    def __init__(self):
        self.stt_available = False
        self.tts_available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check which voice features are available."""
        try:
            import speech_recognition
            self.stt_available = True
        except ImportError:
            pass
        
        try:
            import pyttsx3
            self.tts_available = True
            self.tts_engine = pyttsx3.init()
        except:
            pass
    
    def listen(self, timeout: int = 5) -> str:
        """Listen for voice input and convert to text."""
        if not self.stt_available:
            return ""
        
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = recognizer.listen(source, timeout=timeout)
                text = recognizer.recognize_google(audio)
                print(f"   Heard: {text}")
                return text
            except sr.WaitTimeoutError:
                print("   (No speech detected)")
                return ""
            except sr.UnknownValueError:
                print("   (Could not understand)")
                return ""
            except Exception as e:
                print(f"   Error: {e}")
                return ""
    
    def speak(self, text: str):
        """Convert text to speech."""
        if not self.tts_available:
            print(f"🔊 {text}")
            return
        
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def is_available(self) -> dict:
        """Check what's available."""
        return {
            "speech_to_text": self.stt_available,
            "text_to_speech": self.tts_available,
        }


# To enable voice:
# pip install SpeechRecognition pyttsx3 pyaudio
