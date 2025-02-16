# emotion_detection.py

import os
import torchaudio
import torch
import tempfile
from speechbrain.inference.interfaces import foreign_class

from transcribe import phrases

def detect_emotion(phrase, audio_tensor, sample_rate, classifier):
    """
    Given a phrase dict (with "start" and "end"), extract the corresponding
    audio segment from audio_tensor and use the classifier to predict its emotion.
    Returns the predicted emotion as a full word.
    """
    # Convert start/end times (in seconds) to sample indices
    start_sample = int(phrase["start"] * sample_rate)
    end_sample = int(phrase["end"] * sample_rate)
    segment = audio_tensor[:, start_sample:end_sample]
    
    # If multi-channel, convert to mono by averaging channels.
    if segment.shape[0] > 1:
        segment = segment.mean(dim=0, keepdim=True)
    
    # Save the segment to a temporary WAV file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_filename = tmp.name
    torchaudio.save(temp_filename, segment, sample_rate)
    
    # Use the classifier to get predictions.
    # Returns a tuple: (out_prob, score, index, text_lab)
    # -> we only need the text label the other variables are placeholders and might be used in the future.
    out_prob, score, index, text_lab = classifier.classify_file(temp_filename)
    
    # Remove the temporary file.
    os.remove(temp_filename)
    
    # If text_lab is a list, take its first element.
    if isinstance(text_lab, list):
        text_lab = text_lab[0]
    
    # Clean the label: remove brackets and quotes.
    label_clean = text_lab.strip("[]").replace("'", "").strip()
    
    # Optionally, map abbreviated labels to full words.
    emotion_map = {
        "neu": "neutral",
        "hap": "happy",
        "sad": "sad",
        "ang": "angry",
        "fea": "fearful",
        "dis": "disgusted"
    }
    # Use the mapping if available, otherwise keep the cleaned label.
    emotion = emotion_map.get(label_clean.lower(), label_clean)
    
    return emotion

def add_emotion_to_phrases(phrases, audio_file, device=None):
    """
    Given a list of phrases (each a dict with "start", "end", and "text") and an audio file path.
    Load the audio and for each phrase detect its emotion. 
    The detected emotion is added to the phrase's beginning. 
    Returns the updated list of phrases.
    """
    # Load the audio file once.
    audio_tensor, sample_rate = torchaudio.load(audio_file)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the classifier using foreign_class .
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    
    # Process each phrase -> detect emotion and update its text.
    for phrase in phrases:
        label = detect_emotion(phrase, audio_tensor, sample_rate, classifier)
        phrase["text"] = f"{label} {phrase['text']}"
    
    return phrases


phrases_with_emotion = add_emotion_to_phrases(phrases, "input.wav")

# Print updated phrases.
print("Phrases with Detected Emotion:")
for p in phrases_with_emotion:
    print(p)
