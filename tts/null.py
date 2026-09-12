"""No-op TTS backend for sessions driven by externally supplied PCM."""

from registry import register
from tts.base_tts import BaseTTS


@register("tts", "none")
class NullTTS(BaseTTS):
    """Keep the standard lifecycle while producing no audio of its own."""

    def txt_to_audio(self, msg: tuple[str, dict]):
        return None
