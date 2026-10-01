# -*- coding: utf-8 -*-
"""
Tiện ích xử lý đa phương tiện (Video & Audio):
- Trích xuất luồng âm thanh từ file Video (MP4, MKV, AVI, MOV, WEBM,...) hoặc Audio.
- Chuẩn hóa về 16kHz Mono Float32.
- Xuất phụ đề SRT, WebVTT và TXT từ danh sách phân đoạn người nói.
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def extract_audio_from_media(filepath: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Trích xuất âm thanh từ bất kỳ file Video hoặc Audio nào,
    chuẩn hóa về định dạng Mono Float32 [-1.0, 1.0] với sample rate target_sr (mặc định 16kHz).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: '{filepath}'")

    # Cách 1: Sử dụng PyAV (hỗ trợ mọi định dạng Video MP4, MKV, AVI, MOV, WEBM và Audio MP3, M4A, FLAC, WAV)
    try:
        import av
        container = av.open(filepath)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if audio_streams:
            stream = audio_streams[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            chunks = []
            for frame in container.decode(stream):
                for r_frame in resampler.resample(frame):
                    arr = r_frame.to_ndarray()[0]
                    chunks.append(arr)
            container.close()
            if chunks:
                pcm = np.concatenate(chunks).astype(np.float32)
                return pcm, target_sr
    except Exception as e:
        print(f"[media_utils] PyAV decoding gặp thông báo: {e}, thử phương thức dự phòng...")

    # Cách 2: Dự phòng với soundfile
    try:
        import soundfile as sf
        audio_data, sr = sf.read(filepath, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sr != target_sr:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        return audio_data.astype(np.float32), target_sr
    except Exception as e2:
        print(f"[media_utils] Soundfile decoding gặp thông báo: {e2}, thử librosa...")

    # Cách 3: Dự phòng với librosa
    try:
        import librosa
        audio_data, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio_data.astype(np.float32), target_sr
    except Exception as e3:
        raise RuntimeError(f"Không thể đọc âm thanh từ file '{os.path.basename(filepath)}'. Chi tiết lỗi: {e3}")


def _format_timestamp_srt(seconds: float) -> str:
    """Chuyển giây sang định dạng SRT: HH:MM:SS,mmm"""
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Chuyển giây sang định dạng WebVTT: HH:MM:SS.mmm"""
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: List[Dict[str, Any]]) -> str:
    """Tạo nội dung file phụ đề SRT kèm nhãn người nói."""
    lines = []
    idx = 1
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start_t = _format_timestamp_srt(seg.get("start", 0.0))
        end_t = _format_timestamp_srt(seg.get("end", 0.0))
        spk = seg.get("speaker", "Speaker")

        lines.append(str(idx))
        lines.append(f"{start_t} --> {end_t}")
        lines.append(f"[{spk}]: {text}")
        lines.append("")
        idx += 1

    return "\n".join(lines)


def generate_vtt(segments: List[Dict[str, Any]]) -> str:
    """Tạo nội dung file phụ đề WebVTT."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start_t = _format_timestamp_vtt(seg.get("start", 0.0))
        end_t = _format_timestamp_vtt(seg.get("end", 0.0))
        spk = seg.get("speaker", "Speaker")

        lines.append(f"{start_t} --> {end_t}")
        lines.append(f"<v {spk}>{text}")
        lines.append("")

    return "\n".join(lines)


def generate_txt(segments: List[Dict[str, Any]], final_text: str = "") -> str:
    """Tạo biên bản văn bản TXT chi tiết theo lượt nói."""
    lines = [
        "=================================================================",
        "BIÊN BẢN HỘI THOẠI / PHỤ ĐỀ CHI TIẾT THEO NGƯỜI NÓI",
        "=================================================================",
        ""
    ]
    if segments:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            spk = seg.get("speaker", "Speaker")
            lines.append(f"[{s:6.2f}s -> {e:6.2f}s] {spk}:")
            lines.append(f"  {text}\n")
    elif final_text:
        lines.append(final_text)

    return "\n".join(lines)
