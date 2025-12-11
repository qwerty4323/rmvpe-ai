import ffmpeg
import librosa
import numpy as np


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def detect_breath_mask(audio, sr, hop_length, target_len=None, zcr_threshold=0.25):
    """Return a boolean mask for likely breath/noise frames.

    The mask is computed from frame-wise RMS energy and zero-crossing rate to
    down-weight noisy, non-periodic regions that typically correspond to
    breaths or silence. The length of the returned mask can be forced to
    ``target_len`` to align with downstream f0 arrays.
    """

    frame_length = hop_length * 2
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]

    if rms.size == 0:
        mask = np.zeros(0, dtype=bool)
    else:
        adaptive_floor = max(np.percentile(rms, 25) * 0.6, 1e-4)
        mask = (rms < adaptive_floor) | (zcr > zcr_threshold)

    if target_len is not None:
        if mask.size < target_len:
            pad = np.zeros(target_len - mask.size, dtype=bool)
            mask = np.concatenate([mask, pad])
        else:
            mask = mask[:target_len]

    return mask


def apply_breath_mask_to_f0(f0, audio, sr, hop_length):
    """Zero out f0 values that coincide with breath-heavy regions."""

    mask = detect_breath_mask(audio, sr, hop_length, target_len=len(f0))
    if mask.size and mask.any():
        f0 = f0.copy()
        f0[: mask.size][mask] = 0
    return f0
