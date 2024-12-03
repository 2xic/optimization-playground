import librosa
import numpy as np
import audioread

def load_audio(file_path, sr=None):
    audio, sample_rate = librosa.load(file_path, sr=sr,)
    return audio, sample_rate

# Function to extract features
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    
    feature_vector = ([
        mfcc.mean(), 
        chroma.mean(),
        spectral_contrast.mean(),
        zero_crossing_rate.mean()
    ])
    return feature_vector

def process_audio(input_file, sr=None):
    audio, sample_rate = load_audio(input_file, sr)    
    feature_vector = extract_features(audio, sample_rate)    
    return feature_vector
