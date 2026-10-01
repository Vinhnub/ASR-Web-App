# -*- coding: utf-8 -*-
"""
Module Xác thực & Nhận diện Danh tính Người nói (NVIDIA TitaNet).
Trích xuất Voiceprint 192 chiều và so khớp Cosine Similarity với Database.
Hỗ trợ cơ chế Bật / Tắt (Bypass).
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Speaker_Diarization.speaker_recognition_titanet import TitaNetSpeakerRecognizer, SpeakerDatabase
from Speaker_Diarization.streaming_sortformer import SpeakerSegment


class VerificationModule:
    def __init__(
        self,
        model_name: str = "nvidia/speakerverification_en_titanet_large",
        db_dir: str = "./speaker_db",
        device: str = "cuda",
        threshold: float = 0.55,
        min_speech_duration: float = 0.80,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.db_dir = db_dir
        self.device = device
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration

        self.recognizer: Optional[TitaNetSpeakerRecognizer] = None
        self.db: Optional[SpeakerDatabase] = None

        if self.enabled:
            self._init_model()

    def _init_model(self):
        try:
            if self.model_name.endswith(".onnx"):
                print(f"[VerificationModule] Đang tải TitaNet ONNX '{self.model_name}'...")
                from modules.titanet_onnx import TitaNetONNXRecognizer
                self.recognizer = TitaNetONNXRecognizer(
                    model_path=self.model_name,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
                )
            else:
                print(f"[VerificationModule] Đang tải TitaNet '{self.model_name}' trên {self.device.upper()}...")
                self.recognizer = TitaNetSpeakerRecognizer(
                    model_name=self.model_name,
                    device=self.device
                )
            
            os.makedirs(self.db_dir, exist_ok=True)
            self.db = SpeakerDatabase(
                db_dir=self.db_dir,
                recognizer=self.recognizer
            )
            print(f"[VerificationModule] Khởi tạo TitaNet & Database ({len(self.db.embeddings)} người nói) thành công!")
        except Exception as e:
            print(f"[VerificationModule] Cảnh báo không thể tải TitaNet: {e}")
            self.recognizer = None
            self.db = None

    def resolve_identities(
        self,
        audio_pcm: np.ndarray,
        segments: List[SpeakerSegment],
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> Dict[str, Tuple[str, float]]:
        """
        Nhận diện danh tính thật cho các nhãn phân đoạn từ SortFormer (spk_0, spk_1,...).
        Trả về bản đồ ánh xạ: {raw_speaker: (real_name, score)}.
        Nếu cờ enabled=False, giữ nguyên tên nhãn ban đầu.
        """
        th = threshold if threshold is not None else self.threshold

        # Nếu tắt module: Giữ nguyên nhãn của Diarization
        unique_spks = list(set(s.speaker for s in segments))
        if not self.enabled or self.recognizer is None or self.db is None:
            return {spk: (spk, 1.0) for spk in unique_spks}

        # Trích xuất và ghép các mẩu audio của từng speaker
        speaker_audio_map: Dict[str, List[np.ndarray]] = {}
        total_samples = len(audio_pcm)

        for seg in segments:
            s_idx = max(0, int(seg.start * sample_rate))
            e_idx = min(total_samples, int(seg.end * sample_rate))
            if e_idx > s_idx:
                clip = audio_pcm[s_idx:e_idx]
                if seg.speaker not in speaker_audio_map:
                    speaker_audio_map[seg.speaker] = []
                speaker_audio_map[seg.speaker].append(clip)

        identity_map: Dict[str, Tuple[str, float]] = {}

        for spk, clips in speaker_audio_map.items():
            if not clips:
                identity_map[spk] = (spk, 0.0)
                continue

            full_speaker_clip = np.concatenate(clips, axis=0)
            duration_s = len(full_speaker_clip) / sample_rate

            if duration_s < self.min_speech_duration:
                # Quá ngắn để lấy Voiceprint tin cậy
                identity_map[spk] = (f"{spk} (ngắn)", 0.0)
                continue

            try:
                emb = self.recognizer.extract_embedding(full_speaker_clip, sample_rate=sample_rate)
                matched_name, sim_score = self.db.identify(emb, threshold=th)
                print(f"[VerificationModule] So khớp '{spk}': kết quả='{matched_name}', score={sim_score:.4f}, threshold={th}")

                if matched_name != "Unknown":
                    identity_map[spk] = (matched_name, sim_score)
                else:
                    identity_map[spk] = (f"Chưa đăng ký ({spk})", sim_score)
            except Exception as e:
                print(f"[VerificationModule] Lỗi khi nhận diện {spk}: {e}")
                identity_map[spk] = (spk, 0.0)

        return identity_map

    def identify_single_speaker(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Nhận diện danh tính người nói cho một đoạn âm thanh trực tiếp (One-Shot Realtime).
        Trả về: (display_name, similarity_score)
        """
        th = threshold if threshold is not None else self.threshold
        if not self.enabled or self.recognizer is None or self.db is None or audio_pcm is None or len(audio_pcm) == 0:
            return ("Unknown", 0.0)

        duration_s = len(audio_pcm) / sample_rate
        if duration_s < self.min_speech_duration:
            return ("Unknown", 0.0)

        try:
            emb = self.recognizer.extract_embedding(audio_pcm, sample_rate=sample_rate)
            matched_name, sim_score = self.db.identify(emb, threshold=th)
            if matched_name != "Unknown":
                return (matched_name, float(sim_score))
            else:
                return ("Chưa đăng ký", float(sim_score))
        except Exception as e:
            print(f"[VerificationModule] Lỗi identify_single_speaker: {e}")
            return ("Unknown", 0.0)

    def enroll(self, name: str, audio_pcm: np.ndarray, sample_rate: int = 16000, description: str = ""):
        """Đăng ký hồ sơ giọng nói mới."""
        if self.recognizer is None or self.db is None:
            self._init_model()
        if self.db is not None:
            return self.db.enroll(name, audio_pcm, sample_rate=sample_rate, description=description)
        raise RuntimeError("Speaker database chưa sẵn sàng.")

    def list_enrolled_speakers(self) -> List[Dict]:
        """Danh sách những người đã đăng ký giọng nói."""
        if self.db is None:
            return []
        return list(self.db.metadata.values())

    def delete_speaker(self, name: str) -> bool:
        """Xóa một người nói khỏi Database."""
        if self.db is None:
            return False
        if hasattr(self.db, "delete_speaker"):
            return self.db.delete_speaker(name)
        elif hasattr(self.db, "delete"):
            return self.db.delete(name)
        return False

    def set_parameters(
        self,
        threshold: Optional[float] = None,
        min_speech_duration: Optional[float] = None,
    ):
        """Cập nhật linh hoạt các thông số nhận diện TitaNet."""
        if threshold is not None:
            self.threshold = float(threshold)
        if min_speech_duration is not None:
            self.min_speech_duration = float(min_speech_duration)
        print(f"[VerificationModule] Đã cập nhật tham số: threshold={self.threshold}, min_speech_duration={self.min_speech_duration}s")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if self.enabled and self.recognizer is None:
            self._init_model()
