# feature_audio.py
import librosa
import numpy as np
import subprocess
import os
import uuid


def _convert_to_pcm_wav(input_path, sr=16000):
    """
    Converts any browser-recorded audio to standard PCM WAV
    using ffmpeg (required on macOS).
    """
    output_path = f"./uploads/conv_{uuid.uuid4().hex}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output_path


def extract_prosodic(path, sr=16000):
    """
    Robust prosodic feature extraction:
    - RMS energy
    - pause ratio
    - onset/speaking rate

    Works safely with browser audio (WebM / Opus / WAV).
    """

    try:
        # Try loading directly (works for real PCM WAV)
        y, _ = librosa.load(path, sr=sr)

    except Exception:
        # Fallback: convert to standard PCM WAV
        safe_path = _convert_to_pcm_wav(path, sr=sr)
        y, _ = librosa.load(safe_path, sr=sr)

        # Cleanup converted file
        if os.path.exists(safe_path):
            os.remove(safe_path)

    duration = len(y) / sr

    if duration <= 0:
        return {
            "energy": 0.0,
            "pause_ratio": 0.0,
            "onset_rate": 0.0
        }

    # RMS Energy
    energy = float(np.mean(librosa.feature.rms(y=y)))

    # Pause ratio (non-speech percentage)
    intervals = librosa.effects.split(y, top_db=30)
    total_speech = sum((e - s) for s, e in intervals) / sr
    pause_ratio = float(1.0 - (total_speech / duration))

    # Speaking rate (onsets per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = float(len(onsets) / duration)

    return {
        "energy": round(energy, 6),
        "pause_ratio": round(pause_ratio, 4),
        "onset_rate": round(onset_rate, 4)
    }


if __name__ == "__main__":
    print("✔ Robust audio feature extractor ready.")
