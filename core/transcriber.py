# Import the Whisper library for speech-to-text transcription
import whisper

# Import os module to access environment variables
import os


# Get Whisper model name from environment variable
# If no environment variable is set, default model = "small"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


# Global variable to store the loaded model
# Initially set to None
_model = None


# Function to load the Whisper model
def laod_model():
    
    # Access the global _model variable
    global _model

    # Load the model only if it has not been loaded already
    # This avoids loading the model again and again
    if _model is None:

        print("Loading the Model...")

        # Load Whisper model (tiny, base, small, medium, large)
        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper Model Loaded.")

    # Return loaded model
    return _model


# Function to transcribe a single audio chunk
def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:

    # Load Whisper model
    model = laod_model()

    # Decide task type
    # If translate=True → convert speech to English
    # Else → normal transcription in original language
    task = "translate" if translate else "transcribe"

    # Perform transcription
    result = model.transcribe(
        chunk_path,
        task=task
    )

    # Return only transcribed text
    return result["text"]


# Function to transcribe all audio chunks
def transcribe_all(chunks: list, translate: bool = False) -> str:

    # Store final combined transcript
    full_transcript = ""

    # Loop through each chunk
    for i, chunk in enumerate(chunks):

        # Print current progress
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")

        # Transcribe current chunk
        text = transcribe_chunk(
            chunk,
            translate=translate
        )

        # Append chunk text to final transcript
        full_transcript += text + " "

    print("Transcription is completed")

    # Return full combined transcript
    return full_transcript