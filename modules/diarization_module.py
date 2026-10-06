# -*- coding: utf-8 -*-
"""
Module Phân đoạn Người nói Thời gian thực (NVIDIA Streaming Sortformer).
Phát hiện và phân tách các lượt nói của từng người (spk_0, spk_1, ...).
Hỗ trợ cơ chế Bật / Tắt (Bypass).
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Speaker_Diarization.streaming_sortformer import StreamingSortformerDiarizer, SpeakerSegment
from Speaker_Diarization.nemotron_diarization import NemotronDiarizer


class DiarizationModule:
    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-3-Diarization",
        device: str = "cuda",
        chunk_len: int = 340,
        collar: float = 0.25,
        min_duration: float = 0.20,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self.chunk_len = chunk_len
        self.collar = collar
        self.min_duration = min_duration
        self.diarizer = None

        if self.enabled:
            self._init_model()

    def _init_model(self):
        try:
            print(f"[DiarizationModule] Đang tải mô hình '{self.model_name}' trên {self.device.upper()}...")
            if "nemotron" in self.model_name.lower():
                self.diarizer = NemotronDiarizer(
                    model_name=self.model_name,
                    device=self.device,
                    collar=self.collar,
                    min_duration=self.min_duration,
                )
            else:
                self.diarizer = StreamingSortformerDiarizer(
                    model_name=self.model_name,
                    device=self.device,
                    chunk_len=self.chunk_len,
                    collar=self.collar,
                    min_duration=self.min_duration,
                )
            print(f"[DiarizationModule] Khởi tạo mô hình thành công!")
        except Exception as e:
            print(f"[DiarizationModule] Cảnh báo không thể nạp mô hình: {e}")
            self.diarizer = None

    def diarize_audio(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> List[SpeakerSegment]:
        """
        Phân đoạn người nói trên một đoạn âm thanh numpy float32.
        Nếu cờ enabled=False, trả về 1 phân đoạn duy nhất cho toàn bộ thời lượng.
        """
        duration = len(audio_pcm) / sample_rate if audio_pcm is not None and len(audio_pcm) > 0 else 0.0
        if duration <= 0:
            return []

        # Nếu tắt module Diarization: Trả về 1 speaker mặc định
        if not self.enabled or self.diarizer is None:
            return [SpeakerSegment(start=0.0, end=round(duration, 3), speaker="speaker_0", confidence=1.0)]

        # Lưu audio vào file tạm thời để NeMo Sortformer xử lý
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            sf.write(temp_path, audio_pcm, sample_rate)
            col = self.collar if collar is None else float(collar)
            min_d = self.min_duration if min_duration is None else float(min_duration)
            segments = self.diarizer.diarize_file(temp_path, collar=col, min_duration=min_d, verbose=False)
            if not segments:
                # Nếu không phát hiện thấy phân đoạn nào, fallback về segment mặc định
                segments = [SpeakerSegment(start=0.0, end=round(duration, 3), speaker="speaker_0", confidence=1.0)]
            return segments
        except Exception as e:
            print(f"[DiarizationModule] Lỗi khi diarize audio: {e}")
            return [SpeakerSegment(start=0.0, end=round(duration, 3), speaker="speaker_0", confidence=1.0)]
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def diarize_audio_numpy(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> List[SpeakerSegment]:
        """
        Phân đoạn người nói trực tiếp trong bộ nhớ (Zero Disk I/O) cho luồng Streaming.
        """
        duration = len(audio_pcm) / sample_rate if audio_pcm is not None and len(audio_pcm) > 0 else 0.0
        if duration <= 0:
            return []

        if not self.enabled or self.diarizer is None:
            return [SpeakerSegment(start=0.0, end=round(duration, 3), speaker="speaker_0", confidence=1.0)]

        col = self.collar if collar is None else float(collar)
        min_d = self.min_duration if min_duration is None else float(min_duration)

        try:
            if hasattr(self.diarizer, "diarize_numpy"):
                segs = self.diarizer.diarize_numpy(audio_pcm, sample_rate=sample_rate, collar=col, min_duration=min_d)
                if segs:
                    return segs
            return self.diarize_audio(audio_pcm, sample_rate=sample_rate, collar=col, min_duration=min_d)
        except Exception as e:
            print(f"[DiarizationModule] Lỗi diarize_audio_numpy, fallback: {e}")
            return self.diarize_audio(audio_pcm, sample_rate=sample_rate, collar=col, min_duration=min_d)

    def detect_vad_activity(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        silence_threshold_s: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Nhận diện VAD và điểm dừng phát biểu cho cả Nemotron và SortFormer.
        """
        duration = len(audio_pcm) / sample_rate if audio_pcm is not None and len(audio_pcm) > 0 else 0.0
        if duration <= 0:
            return {
                "is_speaking": False,
                "silence_duration_s": 0.0,
                "last_speech_end": 0.0,
                "segments": [],
                "active_speaker": None
            }

        if hasattr(self.diarizer, "detect_vad_activity"):
            try:
                return self.diarizer.detect_vad_activity(audio_pcm, sample_rate=sample_rate, silence_threshold_s=silence_threshold_s)
            except Exception as e:
                print(f"[DiarizationModule] Lỗi detect_vad_activity từ diarizer: {e}")

        # Fallback chung (SortFormer hoặc khi detect_vad_activity gặp lỗi):
        segs = self.diarize_audio_numpy(audio_pcm, sample_rate=sample_rate)
        if segs:
            last_end = max(s.end for s in segs)
            silence_d = max(0.0, duration - last_end)
            is_spk = (silence_d < silence_threshold_s)
            active_spk = segs[-1].speaker
        else:
            last_end = 0.0
            silence_d = duration
            is_spk = False
            active_spk = None

        return {
            "is_speaking": is_spk,
            "silence_duration_s": round(silence_d, 3),
            "last_speech_end": round(last_end, 3),
            "segments": segs,
            "active_speaker": active_spk
        }

    def set_parameters(
        self,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
        chunk_len: Optional[int] = None,
    ):
        """Cập nhật linh hoạt các thông số phân đoạn SortFormer."""
        if collar is not None:
            self.collar = float(collar)
            if self.diarizer is not None:
                self.diarizer.collar = float(collar)
        if min_duration is not None:
            self.min_duration = float(min_duration)
            if self.diarizer is not None:
                self.diarizer.min_duration = float(min_duration)
        if chunk_len is not None:
            self.chunk_len = int(chunk_len)
            if self.diarizer is not None:
                self.diarizer.chunk_len = int(chunk_len)
        print(f"[DiarizationModule] Đã cập nhật tham số: collar={self.collar}s, min_duration={self.min_duration}s, chunk_len={self.chunk_len}")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if self.enabled and self.diarizer is None:
            self._init_model()
