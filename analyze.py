import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import foreign_class from SpeechBrain to load the classifier via your custom interface.
from speechbrain.inference.interfaces import foreign_class

def detect_emotion_whole(audio_file, device=None):
    """
    Given an audio file path, detect the overall emotion using the pretrained
    emotion recognition model from SpeechBrain.
    
    Returns the detected emotion as a full word (e.g., 'neutral', 'happy').
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the classifier using your custom interface.
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    
    # Use the classifier to analyze the entire audio file.
    # The classifier returns a tuple: (out_prob, score, index, text_lab)
    out_prob, score, index, text_lab = classifier.classify_file(audio_file)
    
    # If text_lab is a list, take the first element.
    if isinstance(text_lab, list):
        text_lab = text_lab[0]
    
    # Clean the returned label: remove extraneous characters.
    label_clean = text_lab.strip("[]").replace("'", "").strip()
    
    # Map abbreviated labels to full words.
    emotion_map = {
        "neu": "neutral",
        "hap": "happy",
        "sad": "sad",
        "ang": "angry",
        "fea": "fearful",
        "dis": "disgusted"
    }
    emotion = emotion_map.get(label_clean.lower(), label_clean)
    return emotion

def plot_mel_spectrogram(audio_file, n_mels=128):
    """
    Loads an audio file, computes its mel spectrogram (in decibels), detects its emotion,
    and displays the spectrogram with the detected emotion in the title.
    
    Parameters:
        audio_file (str): Path to the audio file.
        n_mels (int): Number of mel bands to generate.
    """
    # Load the audio file with its native sampling rate.
    y, sr = librosa.load(audio_file, sr=None)
    
    # Compute the mel spectrogram.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # Convert the mel spectrogram to decibel (dB) units.
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Detect the overall emotion for the whole audio file.
    emotion = detect_emotion_whole(audio_file)
    
    # Plot the spectrogram and include the detected emotion in the title.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title(f"Mel-frequency Spectrogram (Detected emotion: {emotion})")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display the mel spectrogram of an audio file with detected emotion."
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file (e.g., input.wav)"
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=128,
        help="Number of mel bands to generate (default: 128)"
    )
    args = parser.parse_args()
    plot_mel_spectrogram(args.audio_file, n_mels=args.n_mels)
