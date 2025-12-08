#!/usr/bin/env python3
"""
Dheera Voice Chat
Chat with voice input and output on Mac.
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dheera import Dheera, DheeraConfig
from connectors.chat_interface import ChatInterface, ChatSession


class VoiceChat:
    """Voice-enabled chat for Dheera on Mac."""
    
    # Available Mac voices (run `say -v ?` to see all)
    VOICES = {
        "male": "Daniel",      # British male
        "female": "Samantha",  # American female
        "indian": "Rishi",     # Indian English male
    }
    
    def __init__(self, voice: str = "male", rate: int = 180):
        self.voice = self.VOICES.get(voice, voice)
        self.rate = rate  # Words per minute
        self.stt_available = self._check_stt()
        
    def _check_stt(self) -> bool:
        """Check if speech recognition is available."""
        try:
            import speech_recognition
            return True
        except ImportError:
            return False
    
    def speak(self, text: str):
        """Speak text using Mac's say command."""
        try:
            # Clean text for speech (remove special chars)
            clean_text = text.replace('"', "'").replace('\n', ' ')
            
            # Use Mac's built-in say command
            subprocess.run([
                "say",
                "-v", self.voice,
                "-r", str(self.rate),
                clean_text
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"[Voice error: {e}]")
        except FileNotFoundError:
            print("[Voice not available - 'say' command not found]")
    
    def listen(self, timeout: int = 5) -> str:
        """Listen for voice input."""
        if not self.stt_available:
            print("⚠️  Speech recognition not installed.")
            print("   Install: pip install SpeechRecognition pyaudio")
            return ""
        
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (speak now)")
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
            print("   Processing...")
            text = recognizer.recognize_google(audio)
            print(f"   Heard: \"{text}\"")
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
    
    def test_voice(self):
        """Test voice output."""
        print("\n🔊 Testing voice output...")
        self.speak("Hello Nandha! I am Dheera, your AI assistant. Voice is working!")
        print("✅ If you heard that, voice is working!\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dheera Voice Chat")
    parser.add_argument("--voice", "-v", default="male", choices=["male", "female", "indian"])
    parser.add_argument("--rate", "-r", type=int, default=180, help="Speech rate (words/min)")
    parser.add_argument("--test", action="store_true", help="Test voice and exit")
    parser.add_argument("--text-only", action="store_true", help="Type input, hear output")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║       D H E E R A  - Voice Mode 🎤                    ║
    ║       Created by Nandha                               ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Initialize voice
    voice = VoiceChat(voice=args.voice, rate=args.rate)
    
    # Test mode
    if args.test:
        voice.test_voice()
        return
    
    # Initialize Dheera
    print("Initializing Dheera...\n")
    config = DheeraConfig(
        slm_provider="ollama",
        slm_model="phi3:mini",
        debug_mode=args.debug,
    )
    dheera = Dheera(config=config)
    
    # Greeting
    greeting = f"Hello Nandha! I am Dheera. How can I help you today?"
    print(f"\n🤖 Dheera: {greeting}\n")
    voice.speak(greeting)
    
    # Chat loop
    print("Commands: /quit to exit, /voice to toggle voice, /test to test voice\n")
    
    voice_enabled = True
    session = ChatSession(session_id=f"voice_{int(__import__('time').time())}")
    
    while True:
        try:
            # Get input
            if args.text_only or not voice.stt_available:
                user_input = input("You: ").strip()
            else:
                print("\nPress Enter to speak (or type your message):")
                user_input = input("You: ").strip()
                
                if user_input == "":
                    user_input = voice.listen()
                    if not user_input:
                        continue
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                farewell = "Goodbye Nandha! It was nice talking to you."
                print(f"\n🤖 Dheera: {farewell}")
                voice.speak(farewell)
                break
            
            if user_input.lower() == "/voice":
                voice_enabled = not voice_enabled
                status = "enabled" if voice_enabled else "disabled"
                print(f"🔊 Voice output {status}")
                continue
            
            if user_input.lower() == "/test":
                voice.test_voice()
                continue
            
            # Get response from Dheera
            response = dheera.process_message(user_input, session)
            
            # Display and speak
            print(f"\n🤖 Dheera: {response}\n")
            
            if voice_enabled:
                # Truncate for speech if too long
                speech_text = response[:500] if len(response) > 500 else response
                voice.speak(speech_text)
            
            # Show action in debug mode
            if args.debug:
                action = dheera.last_action_info.get("action_name", "?")
                epsilon = dheera.last_action_info.get("epsilon", 0)
                print(f"   [Action: {action}, ε: {epsilon:.3f}]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving...")
            dheera.save_checkpoint()
            break
    
    dheera.end_conversation()
    print("\n👋 Session ended.\n")


if __name__ == "__main__":
    main()
