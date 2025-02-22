#to play the audio i . e text to audio
from gtts import gTTS
import pygame
import os
import time

def play_text_to_speech(text, language='en', slow=False): #since we don't want to play the audio slowly
    #latency measurement
    start_time = time.perf_counter()  # Start timer for TTS

    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    tts_time = time.perf_counter() - start_time  # End timer for TTS
    print(f"TTS Generation Time: {tts_time:.3f} seconds")

    # Start timer for audio playback
    playback_start_time = time.perf_counter()

    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed
    
    playback_time = time.perf_counter() - playback_start_time  # End timer for playback
    print(f"Audio Playback Time: {playback_time:.3f} seconds")

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)
