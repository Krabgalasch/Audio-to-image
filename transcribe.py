from faster_whisper import WhisperModel
import re


def transcribe_and_split(
    model_path="small", audio_path="input.wav", pause_threshold=0.5
):
    """
    Transcribe audio and split into shorter sentences, considering pauses of 0.5 seconds and sentence-ending punctuation.

    Parameters:
        model_path (str): Path to the FasterWhisper model.
        audio_path (str): Path to the audio file.
        pause_threshold (float): Minimum pause duration (in seconds) to consider as a new phrase.

    Returns:
        List of sentences with text, start, and end times.
    """
    # Load Faster Whisper model
    model = WhisperModel(model_path, device="cuda")
    segments, _ = model.transcribe(audio_path)

    phrases = []
    previous_end = 0.0

    for segment in segments:
        # Check if there's a significant pause
        if segment.start - previous_end >= pause_threshold:
            # Start a new phrase
            phrases.append(
                {
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                }
            )
        else:
            # Merge with the previous phrase
            if phrases:
                phrases[-1]["text"] += f" {segment.text.strip()}"
                phrases[-1]["end"] = segment.end
            else:
                # First segment
                phrases.append(
                    {
                        "text": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                    }
                )

        previous_end = segment.end

    # Split merged phrases into sentences
    split_phrases = []
    for phrase in phrases:
        sentences = re.split(r"(?<=[,.!?])\s+", phrase["text"])  # Split on punctuation
        total_duration = phrase["end"] - phrase["start"]
        num_sentences = len(sentences)

        # Divide time proportionally for each sentence
        sentence_start = phrase["start"]
        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty sentences

            duration = total_duration / num_sentences
            sentence_end = sentence_start + duration

            split_phrases.append(
                {"text": sentence.strip(), "start": sentence_start, "end": sentence_end}
            )
            sentence_start = sentence_end

    return split_phrases


# Execute the function for transcribing and splitting
phrases = transcribe_and_split(
    model_path="small", audio_path="input.wav", pause_threshold=0.3
)

# Debug Output
print("Processed Phrases:")
for p in phrases:
    print(p)
