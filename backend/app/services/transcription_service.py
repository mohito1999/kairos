import asyncio
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions

from app.core.config import settings

class TranscriptionService:
    def __init__(self):
        self.client = DeepgramClient(settings.DEEPGRAM_API_KEY)

    async def transcribe_audio_from_url(self, audio_url: str) -> str:
        """
        Transcribes audio from a public URL using Deepgram Nova-2 with diarization.
        Returns a formatted transcript string.
        """
        try:
            source = {"url": audio_url}
            # Options to enable diarization and smart formatting (punctuation, etc.)
            options = PrerecordedOptions(
                model="nova-3",
                smart_format=True,
                diarize=True,
            )

            response = await self.client.listen.prerecorded.v("1").transcribe_url(
                source, options
            )
            
            # Format the diarized transcript into a human-readable string
            transcript = ""
            for paragraph in response.results.paragraphs.paragraphs:
                speaker = paragraph.speaker
                text = " ".join(sentence.text for sentence in paragraph.sentences)
                transcript += f"Speaker {speaker}: {text}\n"
            
            return transcript.strip()

        except Exception as e:
            print(f"Error during Deepgram transcription: {e}")
            return f"Error transcribing audio: {e}"

# Create a singleton instance to be used across the application
transcription_service = TranscriptionService()