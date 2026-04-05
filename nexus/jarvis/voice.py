"""
Jarvis Voice Layer — Whisper STT + Coqui TTS.
Handles microphone input and voice output.
"""

import os
import sys
import tempfile
from rich.console import Console

console = Console()


class VoiceInput:
    """
    Speech-to-text using faster-whisper.
    Listens to microphone and transcribes speech.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        """Lazy-load Whisper model."""
        if not self._model:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                console.print(f"[green]Whisper {self.model_size} model loaded[/green]")
            except ImportError:
                console.print("[yellow]faster-whisper not installed. Run: pip install faster-whisper[/yellow]")
                return None
        return self._model

    def listen(self, duration: int = 5) -> str:
        """
        Listen to microphone for specified duration and transcribe.

        Args:
            duration: seconds to listen

        Returns:
            Transcribed text string
        """
        try:
            import sounddevice as sd
            import numpy as np
            import scipy.io.wavfile as wav

            console.print(f"[cyan]Listening for {duration} seconds...[/cyan]")
            sample_rate = 16000
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav.write(f.name, sample_rate, recording)
                temp_path = f.name

            model = self._get_model()
            if not model:
                return ""

            segments, _ = model.transcribe(temp_path, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
            os.unlink(temp_path)
            console.print(f"[dim]Heard: {text}[/dim]")
            return text

        except Exception:
            console.print("[yellow]Voice input is unavailable right now. Check your microphone permissions and audio dependencies.[/yellow]")
            return ""


class VoiceOutput:
    """
    Text-to-speech using Coqui TTS.
    Converts text responses to spoken audio.
    """

    def __init__(self):
        self._tts = None

    def _get_tts(self):
        """Lazy-load Coqui TTS."""
        if not self._tts:
            try:
                from TTS.api import TTS
                self._tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                console.print("[green]Coqui TTS loaded[/green]")
            except ImportError:
                console.print("[yellow]TTS not installed. Run: pip install TTS[/yellow]")
                return None
        return self._tts

    def speak(self, text: str):
        """
        Convert text to speech and play it.

        Args:
            text: text to speak
        """
        tts = self._get_tts()
        if not tts:
            console.print(f"[bold]{text}[/bold]")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            tts.tts_to_file(text=text[:500], file_path=temp_path)

            if sys.platform.startswith("win"):
                import winsound

                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
            else:
                import subprocess

                subprocess.run(["aplay", temp_path], capture_output=True, check=False)
            os.unlink(temp_path)

        except Exception:
            console.print("[yellow]Voice output is unavailable right now. Falling back to text.[/yellow]")
            console.print(f"[bold]NEXUS:[/bold] {text}")
