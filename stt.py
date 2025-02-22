#Load credentials
import os
import time
from dotenv import load_dotenv
import librosa
#to record audios
import wave
import pyaudio  # audio in real-time
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

#checking if the audio is silent. If ever the amplitude is less than threshold we consider it to be silent and hence we will not process it
def is_silence(data, max_amplitude_threshold=500):  # Lowered threshold from 3000 to 500
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    print(f"Max amplitude detected: {max_amplitude}")  # Debug: log max amplitude
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=5):  # audio will record 5 sec
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    #here, we start appending the frames of audio, chunk by chunk
    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)  # Changed from 4096 to 1024 for consistency with frames_per_buffer
        frames.append(data)

    print("Processing in-memory...")
    raw_data = b''.join(frames)
    # Convert binary data to a NumPy array (int16)
    audio_np = np.frombuffer(raw_data, dtype=np.int16)

    # Check if the recorded chunk contains silence; if so, skip processing
    if is_silence(audio_np):
        print("Silent audio detected; skipping processing.")
        return None
    else:
        return audio_np

#loading the model
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    return processor, model

# Function to transcribe audio with latency tracking
def transcribe_audio(processor, model, audio_array):
    start_time = time.perf_counter()  # Start timer
    # Normalize the audio: convert int16 to float32 in range [-1, 1]
    audio_input = audio_array.astype('float32') / 32768.0
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    stt_time = time.perf_counter() - start_time  # End timer
    print(f"STT Time Taken: {stt_time:.3f} seconds")  # Log latency
    return transcription.strip()
