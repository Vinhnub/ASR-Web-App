# -*- coding: utf-8 -*-
"""
Cấu hình tập trung cho ứng dụng Realtime ASR Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict

# Đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = BASE_DIR


@dataclass
class PipelineConfig:
    # 1. Các cờ Bật / Tắt từng khối (Toggle flags)
    enable_denoise: bool = False
    enable_asr: bool = True
    enable_diarization: bool = True
    enable_verification: bool = True
    enable_hotword: bool = True
    enable_punctuation: bool = True
    enable_summary: bool = True  # Bật/tắt tóm tắt nội dung tự động (Qwen2.5-1.5B)
    enable_realtime_diarization: bool = True  # Bật/tắt phân đoạn & nhận diện người nói thời gian thực
    realtime_diarization_interval_s: float = 2.0  # Chu kỳ cập nhật Diarization tối ưu hiệu năng (giây)
    max_recent_words_per_speaker: int = 100  # Số từ gần nhất hiển thị cho mỗi người nói (đồng bộ với lượt nói đầy đủ)
    speaker_inactivity_timeout_s: float = 7.0  # Thời gian ẩn người không nói (giây)
    enable_2pass_streaming: bool = True  # Chốt lượt nói cuốn chiếu bằng ZipFormer-150M
    turn_silence_timeout_s: float = 0.7  # Thời gian im lặng (s) để chốt lượt nói của chunk
    turn_min_speech_s: float = 0.4  # Thời lượng nói tối thiểu (s) để kích hoạt ZipFormer-150M

    # 2. Thiết bị phần cứng
    device: str = "cuda"  # 'cuda' hoặc 'cpu'

    # 3. Cấu hình âm thanh
    sample_rate: int = 16000
    chunk_size_s: float = 0.32  # 320ms mỗi chunk streaming (5120 samples tại 16kHz)
    
    # 4. Cấu hình Denoise (DeepFilterNet3)
    denoise_model: str = "DeepFilterNet3"
    denoise_atten_lim_db: float = 15.0  # 15.0 dB: Khử sạch tạp âm nhưng bảo toàn 100% âm thanh đầu khi người mới nói

    # 5. Cấu hình ASR (ZipFormer Streaming) & In-Decoding Hotwords
    asr_model_dir: str = os.path.join(BASE_DIR, "zipformer_streaming_vi")
    asr_threads: int = 4
    asr_provider: str = "cpu"  # 'cpu' hoặc 'cuda' (CPU nhẹ và tiết kiệm GPU VRAM)
    asr_decoding_method: str = "modified_beam_search"
    asr_max_active_paths: int = 4
    asr_hotwords_score: float = 2.0
    asr_bpe_vocab: str = os.path.join(BASE_DIR, "zipformer_streaming_vi", "bpe_vocab.txt")

    # 6. Cấu hình Diarization (Nemotron-3)
    diarization_model: str = "nvidia/Nemotron-3-Diarization"
    diarization_chunk_len: int = 340
    diarization_collar: float = 0.25
    diarization_min_duration: float = 0.20

    # 7. Cấu hình Verification & Voiceprint Tracking (TitaNet)
    verification_model: str = "nvidia/speakerverification_en_titanet_large"
    speaker_db_dir: str = os.path.join(APP_DIR, "speaker_db")
    verification_threshold: float = 0.75
    min_speech_duration_for_id: float = 0.80
    speaker_tracking_threshold: float = 0.55  # Ngưỡng Cosine Similarity để theo dõi khóa danh tính trong phiên

    # 8. Cấu hình Hotwords (Contextual Biasing trực tiếp trong ZipFormer)
    hotwords_file: str = os.path.join(APP_DIR, "hotwords.txt")

    # 9. Cấu hình Automatic Punctuation (ViBERT-capu)
    punctuation_variant: str = "int8"  # 'int8' hoặc 'fp32'

    # 10. Cấu hình AI Summarization (Qwen2.5-1.5B-Instruct)
    summary_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    summary_max_tokens: int = 384
    summary_temperature: float = 0.3


    # 11. Cấu hình Server
    host: str = "0.0.0.0"
    port: int = 8000

    def to_dict(self) -> Dict:
        return {
            # Bật / Tắt modules
            "enable_denoise": self.enable_denoise,
            "enable_asr": self.enable_asr,
            "enable_diarization": self.enable_diarization,
            "enable_verification": self.enable_verification,
            "enable_hotword": self.enable_hotword,
            "enable_punctuation": self.enable_punctuation,
            "enable_summary": self.enable_summary,
            "enable_realtime_diarization": self.enable_realtime_diarization,
            "realtime_diarization_interval_s": self.realtime_diarization_interval_s,
            "max_recent_words_per_speaker": self.max_recent_words_per_speaker,
            "speaker_inactivity_timeout_s": self.speaker_inactivity_timeout_s,
            "enable_2pass_streaming": self.enable_2pass_streaming,
            "turn_silence_timeout_s": self.turn_silence_timeout_s,
            "turn_min_speech_s": self.turn_min_speech_s,
            # Cấu hình phần cứng & Audio
            "device": self.device,
            "sample_rate": self.sample_rate,
            "chunk_size_s": self.chunk_size_s,
            # TitaNet Speaker Verification
            "verification_threshold": self.verification_threshold,
            "min_speech_duration_for_id": self.min_speech_duration_for_id,
            "speaker_tracking_threshold": self.speaker_tracking_threshold,
            # Speaker Diarization
            "diarization_model": self.diarization_model,
            "diarization_collar": self.diarization_collar,
            "diarization_min_duration": self.diarization_min_duration,
            "diarization_chunk_len": self.diarization_chunk_len,
            # ZipFormer ASR & Hotwords
            "asr_decoding_method": self.asr_decoding_method,
            "asr_max_active_paths": self.asr_max_active_paths,
            "asr_hotwords_score": self.asr_hotwords_score,
            # DeepFilterNet3
            "denoise_atten_lim_db": self.denoise_atten_lim_db,
            # ViBERT-capu Punctuation
            "punctuation_variant": self.punctuation_variant,
            # Qwen2.5-1.5B Summarization
            "summary_model": self.summary_model,
            "summary_max_tokens": self.summary_max_tokens,
            "summary_temperature": self.summary_temperature,
        }

    def update(self, new_params: Dict) -> None:
        """Cập nhật các thông số từ dictionary với ép kiểu an toàn."""
        for k, v in new_params.items():
            if hasattr(self, k) and v is not None:
                current_val = getattr(self, k)
                try:
                    if isinstance(current_val, bool):
                        setattr(self, k, str(v).lower() in ("true", "1", "yes"))
                    elif isinstance(current_val, int):
                        setattr(self, k, int(v))
                    elif isinstance(current_val, float):
                        setattr(self, k, float(v))
                    else:
                        setattr(self, k, v)
                except Exception as e:
                    print(f"[PipelineConfig] Lỗi ép kiểu tham số '{k}': {e}")
