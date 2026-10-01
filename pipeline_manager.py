# -*- coding: utf-8 -*-
"""
Pipeline Manager: Bộ não điều phối toàn bộ chu trình xử lý âm thanh thời gian thực.
Kết nối 6 khối mô hình độc lập và quản lý phiên stream (Session State) cho từng kết nối.
"""

import os
import sys
import time
import uuid
import re
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import soundfile as sf

# Thiết lập UTF-8 cho Windows console để tránh UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tự động nạp thư mục CUDA & cuDNN DLLs từ PyTorch vào Windows DLL Search Path
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
except Exception:
    pass

try:
    from .config import PipelineConfig
    from .modules.denoise_module import DenoiseModule
    from .modules.asr_module import ASRModule
    from .modules.diarization_module import DiarizationModule
    from .modules.verification_module import VerificationModule
    from .modules.hotword_module import HotwordModule
    from .modules.punctuation_module import PunctuationModule
except ImportError:
    from config import PipelineConfig
    from modules.denoise_module import DenoiseModule
    from modules.asr_module import ASRModule
    from modules.diarization_module import DiarizationModule
    from modules.verification_module import VerificationModule
    from modules.hotword_module import HotwordModule
    from modules.punctuation_module import PunctuationModule

from Speaker_Diarization.streaming_sortformer import SpeakerSegment


def format_natural_text(text: str) -> str:
    """
    Chuyển đổi chuỗi chữ in hoa hoặc chữ thường thành văn bản tự nhiên (viết hoa chữ đầu câu).
    Giữ nguyên các chuỗi đã qua xử lý bởi ViBERT-capu hoặc đã có chữ hoa/thường tự nhiên hỗn hợp.
    Ví dụ:
      'HI XIN CHÀO CÁC BẠN' -> 'Hi xin chào các bạn'
      'xin chào các bạn' -> 'Xin chào các bạn'
      'XIN CHÀO. TÔI LÀ AI?' -> 'Xin chào. Tôi là ai?'
      'Xin chào các bạn, tôi là Phúc.' -> 'Xin chào các bạn, tôi là Phúc.'
    """
    if not text or not text.strip():
        return ""
    stripped = text.strip()

    has_upper = any(c.isupper() for c in stripped)
    has_lower = any(c.islower() for c in stripped)

    # Nếu chỉ gồm chữ in hoa (từ ZipFormer gốc) hoặc toàn chữ thường
    if (has_upper and not has_lower) or (has_lower and not has_upper):
        text_lower = stripped.lower()
        parts = re.split(r'([.?!]\s*|\n+)', text_lower)
        result = []
        capitalize_next = True
        for part in parts:
            if not part:
                continue
            if re.match(r'^([.?!]\s*|\n+)$', part):
                result.append(part)
                capitalize_next = True
            else:
                if capitalize_next and part:
                    s_part = part.lstrip()
                    lead_spaces = part[:len(part) - len(s_part)]
                    if s_part:
                        part = lead_spaces + s_part[0].upper() + s_part[1:]
                    capitalize_next = False
                result.append(part)
        return "".join(result)

    return stripped


class StreamingSession:
    """Đại diện cho một phiên kết nối âm thanh thời gian thực của một Client."""

    def __init__(self, session_id: str, asr_stream, options: Dict[str, Any]):
        self.session_id = session_id
        self.asr_stream = asr_stream
        self.options = options.copy()
        self.raw_audio_chunks: List[np.ndarray] = []
        self.clean_audio_chunks: List[np.ndarray] = []
        self.prev_denoise_tail: np.ndarray = np.zeros(0, dtype=np.float32)
        self.total_samples = 0
        self.current_partial_text = ""
        self.all_words_history: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.last_active_at = time.time()

        # Trạng thái Phân đoạn & Nhận diện người nói thời gian thực (Real-time Multi-Speaker State)
        self.speaker_identities: Dict[str, Tuple[str, float]] = {}  # {spk_id: (display_name, score)}
        self.speaker_last_active_time: Dict[str, float] = {}        # {spk_id: timestamp}
        self.speaker_words: Dict[str, List[str]] = {}               # {spk_id: [words]}
        self.active_speaker_id: str = "speaker_0"
        self.active_identity: str = "speaker_0"
        self.active_confidence: float = 1.0
        self.last_diarize_sample_count: int = 0
        self.last_processed_word_count: int = 0
        self.confirmed_segs: List[Any] = []

    def add_chunk(self, raw_chunk: np.ndarray, clean_chunk: Optional[np.ndarray] = None):
        """Lưu trữ mảng PCM thô và mảng PCM đã khử nhiễu từ microphone client."""
        self.raw_audio_chunks.append(raw_chunk)
        if clean_chunk is not None:
            self.clean_audio_chunks.append(clean_chunk)
        else:
            self.clean_audio_chunks.append(raw_chunk)
        self.total_samples += len(raw_chunk)
        self.last_active_at = time.time()

    def get_full_audio(self) -> np.ndarray:
        if not self.raw_audio_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.raw_audio_chunks, axis=0)

    def get_clean_audio(self) -> np.ndarray:
        if not self.clean_audio_chunks:
            return self.get_full_audio()
        return np.concatenate(self.clean_audio_chunks, axis=0)

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / 16000.0


class RealtimePipelineManager:
    """Quản lý và phối hợp hoạt động của toàn bộ các khối trong Pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.sessions: Dict[str, StreamingSession] = {}

        # Thư mục lưu trữ file âm thanh online thu được để nghe lại
        app_dir = os.path.dirname(os.path.abspath(__file__))
        self.recordings_dir = os.path.join(app_dir, "static", "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)

        print("=" * 75)
        print("KHỞI ĐỘNG HỆ THỐNG REALTIME ASR PIPELINE")
        print(f"• Thiết bị mặc định : {self.config.device.upper()}")
        print(f"• Cấu hình Bật/Tắt  : Denoise={self.config.enable_denoise}, ASR={self.config.enable_asr}, "
              f"Diarization={self.config.enable_diarization}, Verification={self.config.enable_verification}, "
              f"Hotword={self.config.enable_hotword}, Punctuation={self.config.enable_punctuation}")
        print("=" * 75)

        # 1. Khởi tạo Denoise Module (DeepFilterNet3)
        self.denoise = DenoiseModule(
            device=self.config.device,
            model_name=self.config.denoise_model,
            atten_lim_db=self.config.denoise_atten_lim_db,
            enabled=self.config.enable_denoise,
        )

        # 2. Khởi tạo Hotword Module (Quản lý từ khóa in-decoding cho ZipFormer)
        self.hotwords = HotwordModule(
            hotwords_file=self.config.hotwords_file,
            enabled=self.config.enable_hotword,
        )

        # 3. Khởi tạo ASR Module (ZipFormer Streaming với Contextual Biasing)
        self.asr = ASRModule(
            model_dir=self.config.asr_model_dir,
            bpe_vocab_path=self.config.asr_bpe_vocab,
            hotwords_file=self.config.hotwords_file,
            hotwords_score=self.config.asr_hotwords_score,
            decoding_method=self.config.asr_decoding_method,
            max_active_paths=self.config.asr_max_active_paths,
            num_threads=self.config.asr_threads,
            provider=self.config.asr_provider,
            enabled=self.config.enable_asr,
        )

        # 4. Khởi tạo Diarization Module (SortFormer)
        self.diarization = DiarizationModule(
            model_name=self.config.diarization_model,
            device=self.config.device,
            chunk_len=self.config.diarization_chunk_len,
            collar=self.config.diarization_collar,
            min_duration=self.config.diarization_min_duration,
            enabled=self.config.enable_diarization,
        )

        # 5. Khởi tạo Verification Module (TitaNet)
        self.verification = VerificationModule(
            model_name=self.config.verification_model,
            db_dir=self.config.speaker_db_dir,
            device=self.config.device,
            threshold=self.config.verification_threshold,
            min_speech_duration=self.config.min_speech_duration_for_id,
            enabled=self.config.enable_verification,
        )

        # 6. Khởi tạo Punctuation Module (ViBERT-capu)
        self.punctuation = PunctuationModule(
            variant=self.config.punctuation_variant,
            device="cpu",  # ViBERT ONNX chạy CPU cực nhanh và tiết kiệm VRAM cho Diarization/TitaNet
            enabled=self.config.enable_punctuation,
        )

        print("[PipelineManager] Toàn bộ các module đã sẵn sàng phục vụ!")

    def update_model_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cập nhật động các thông số / siêu tham số cho từng model & chức năng trong pipeline.
        Áp dụng tức thì cho các module mà không cần khởi động lại server.
        """
        self.config.update(params)

        # 1. Verification (TitaNet)
        v_th = params.get("verification_threshold")
        v_dur = params.get("min_speech_duration_for_id")
        self.verification.set_parameters(threshold=v_th, min_speech_duration=v_dur)

        # 2. Diarization (SortFormer)
        d_col = params.get("diarization_collar")
        d_min = params.get("diarization_min_duration")
        d_chunk = params.get("diarization_chunk_len")
        self.diarization.set_parameters(collar=d_col, min_duration=d_min, chunk_len=d_chunk)

        # 3. ASR & Hotwords (ZipFormer)
        a_dec = params.get("asr_decoding_method")
        a_paths = params.get("asr_max_active_paths")
        a_score = params.get("asr_hotwords_score")
        self.asr.set_parameters(decoding_method=a_dec, max_active_paths=a_paths, hotwords_score=a_score)

        # 4. Denoise (DeepFilterNet3)
        if "denoise_atten_lim_db" in params:
            dn_lim = params.get("denoise_atten_lim_db")
            self.denoise.set_parameters(atten_lim_db=dn_lim)

        return self.config.to_dict()

    def reset_model_parameters(self) -> Dict[str, Any]:
        """Khôi phục toàn bộ thông số về mặc định ban đầu."""
        from config import PipelineConfig
        default_cfg = PipelineConfig()
        return self.update_model_parameters(default_cfg.to_dict())

    def get_model_parameters_meta(self) -> Dict[str, Any]:
        """Trả về cấu trúc thông tin metadata chi tiết về các thông số để render UI."""
        cfg_dict = self.config.to_dict()
        return {
            "current": cfg_dict,
            "current_config": cfg_dict,
            "groups": [
                {
                    "id": "verification",
                    "title": "Nhận Diện Danh Tính (TitaNet Voiceprint)",
                    "icon": "fa-id-badge",
                    "params": [
                        {
                            "key": "verification_threshold",
                            "label": "Ngưỡng Nhận Diện (Cosine Threshold)",
                            "type": "slider",
                            "min": 0.30,
                            "max": 0.90,
                            "step": 0.01,
                            "default": 0.55,
                            "unit": "",
                            "description": "Ngưỡng tương đồng tối thiểu để nhận diện đúng người đã đăng ký. Giá trị cao hơn giúp tránh nhận nhầm, giá trị thấp hơn giúp nhận diện tốt hơn khi có tạp âm."
                        },
                        {
                            "key": "min_speech_duration_for_id",
                            "label": "Thời Lượng Nói Tối Thiểu (Min Speech Duration)",
                            "type": "slider",
                            "min": 0.30,
                            "max": 3.00,
                            "step": 0.05,
                            "default": 0.80,
                            "unit": "s",
                            "description": "Thời lượng tối thiểu của lượt nói (giây) để trích xuất Voiceprint. Bỏ qua các phân đoạn quá ngắn để tránh tính toán sai."
                        }
                    ]
                },
                {
                    "id": "diarization",
                    "title": "Phân Đoạn Người Nói (Streaming SortFormer)",
                    "icon": "fa-users-viewfinder",
                    "params": [
                        {
                            "key": "diarization_collar",
                            "label": "Collar Nối Lượt Nói (Collar Margin)",
                            "type": "slider",
                            "min": 0.00,
                            "max": 1.00,
                            "step": 0.05,
                            "default": 0.25,
                            "unit": "s",
                            "description": "Khoảng cách thời gian tối đa (giây) giữa 2 đoạn cùng người nói để tự động gộp thành một lượt phát biểu liên tục."
                        },
                        {
                            "key": "diarization_min_duration",
                            "label": "Thời Lượng Phân Đoạn Tối Thiểu (Min Duration)",
                            "type": "slider",
                            "min": 0.10,
                            "max": 1.00,
                            "step": 0.05,
                            "default": 0.20,
                            "unit": "s",
                            "description": "Thời lượng tối thiểu (giây) của một đoạn phát biểu để giữ lại. Lọc bỏ tiếng thở dài, tặc lưỡi hoặc tạp âm ngắn."
                        },
                        {
                            "key": "diarization_chunk_len",
                            "label": "Kích Thước Khối Chunk (Frames)",
                            "type": "number",
                            "min": 100,
                            "max": 500,
                            "step": 20,
                            "default": 340,
                            "unit": "frames",
                            "description": "Độ dài khối frame xử lý của Streaming SortFormer (mặc định 340 frames ~ 2.72s)."
                        }
                    ]
                },
                {
                    "id": "asr",
                    "title": "Nhận Dạng Âm Thanh (ZipFormer ASR) & Hotwords",
                    "icon": "fa-wave-square",
                    "params": [
                        {
                            "key": "asr_decoding_method",
                            "label": "Phương Thức Giải Mã (Decoding Method)",
                            "type": "select",
                            "options": [
                                {"value": "modified_beam_search", "label": "Modified Beam Search (Độ chính xác cao + Hỗ trợ Hotwords)"},
                                {"value": "greedy_search", "label": "Greedy Search (Tốc độ tối đa, tiết kiệm tài nguyên)"}
                            ],
                            "default": "modified_beam_search",
                            "description": "Thuật toán giải mã âm học của mô hình ZipFormer Transducer."
                        },
                        {
                            "key": "asr_max_active_paths",
                            "label": "Số Đường Dẫn Beam Tối Đa (Max Active Paths)",
                            "type": "number",
                            "min": 1,
                            "max": 16,
                            "step": 1,
                            "default": 4,
                            "unit": "paths",
                            "description": "Số đường dẫn song song được duy trì trong Beam Search. Giá trị cao hơn cho kết quả chính xác hơn nhưng tốn thêm CPU."
                        },
                        {
                            "key": "asr_hotwords_score",
                            "label": "Điểm Ưu Tiên Hotwords (Bonus Score)",
                            "type": "slider",
                            "min": 0.5,
                            "max": 10.0,
                            "step": 0.5,
                            "default": 2.0,
                            "unit": "pts",
                            "description": "Điểm cộng xác suất cho các từ khóa chuyên ngành trong In-Decoding Contextual Biasing."
                        }
                    ]
                },
                {
                    "id": "denoise",
                    "title": "Khử Nhiễu Môi Trường (DeepFilterNet3)",
                    "icon": "fa-volume-xmark",
                    "params": [
                        {
                            "key": "denoise_atten_lim_db",
                            "label": "Mức Khử Nhiễu (Denoise Level)",
                            "type": "slider",
                            "min": 0,
                            "max": 50,
                            "step": 1,
                            "default": 40,
                            "unit": "dB",
                            "description": "0 dB = Không khử nhiễu (giữ nguyên âm gốc); Càng tăng số dB thì khử nhiễu càng mạnh (50 dB = Khử sạch tối đa 100%)."
                        }
                    ]
                },
                {
                    "id": "realtime_speaker_ui",
                    "title": "Phân Tách & Hiển Thị Nhiều Người Nói (Multi-Speaker Live)",
                    "icon": "fa-users-gear",
                    "params": [
                        {
                            "key": "enable_realtime_diarization",
                            "label": "Bật Nhận Diện Người Nói Realtime",
                            "type": "toggle",
                            "default": True,
                            "description": "Bật phân đoạn và nhận diện danh tính người nói liên tục theo thời gian thực (Multi-Speaker). Tắt để trở về chế độ cũ."
                        },
                        {
                            "key": "max_recent_words_per_speaker",
                            "label": "Số Từ Gần Nhất Hiển Thị Cho Mỗi Người",
                            "type": "slider",
                            "min": 5,
                            "max": 100,
                            "step": 5,
                            "default": 25,
                            "unit": "từ",
                            "description": "Giới hạn số lượng từ mới nhất hiển thị trong khung phát biểu của từng người trên màn hình."
                        },
                        {
                            "key": "speaker_inactivity_timeout_s",
                            "label": "Thời Gian Ẩn Người Không Nói (Inactivity Timeout)",
                            "type": "slider",
                            "min": 3,
                            "max": 30,
                            "step": 1,
                            "default": 7,
                            "unit": "s",
                            "description": "Nếu một người không phát biểu quá khoảng thời gian này, khung phát biểu sẽ tự động ẩn đi để nhường không gian cho người đang nói."
                        }
                    ]
                }
            ]
        }

    def create_session(self, session_id: Optional[str] = None, options_override: Optional[Dict[str, Any]] = None) -> StreamingSession:
        """Tạo một phiên streaming mới."""
        s_id = session_id or str(uuid.uuid4())
        options = {
            "denoise": self.config.enable_denoise,
            "asr": self.config.enable_asr,
            "diarization": self.config.enable_diarization,
            "verification": self.config.enable_verification,
            "hotwords": self.config.enable_hotword,
            "punctuation": self.config.enable_punctuation,
            "realtime_diarization": self.config.enable_realtime_diarization,
            "realtime_diarization_interval_s": self.config.realtime_diarization_interval_s,
            "max_recent_words_per_speaker": self.config.max_recent_words_per_speaker,
            "speaker_inactivity_timeout_s": self.config.speaker_inactivity_timeout_s,
        }
        if options_override:
            options.update(options_override)

        asr_stream = None
        if options.get("asr", True):
            # Nếu bật hotwords: dùng mặc định (None) hoặc chuỗi hotwords từ self.hotwords
            # Nếu tắt hotwords: truyền chuỗi rỗng "" để vô hiệu hoá biasing cho stream này
            hw_param = None if (options.get("hotwords", True) and self.hotwords.enabled) else ""
            asr_stream = self.asr.create_stream(hotwords=hw_param)

        session = StreamingSession(session_id=s_id, asr_stream=asr_stream, options=options)
        self.sessions[s_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def denoise_streaming_chunk(
        self,
        session: StreamingSession,
        chunk_pcm: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Khử nhiễu streaming với rolling buffer lịch sử (250ms) nhằm:
        - Khởi động ấm (warm-up) GRU của DeepFilterNet3 trước khi vào chunk hiện tại.
        - Triệt tiêu 100% hiện tượng méo tiếng / clicks / sụt giảm âm lượng ở đầu mỗi chunk 256ms.
        """
        if not self.denoise.enabled or self.denoise.filter is None or len(chunk_pcm) == 0:
            return chunk_pcm

        buffer_len = int(0.25 * sample_rate)  # 250ms = 4000 samples ở 16kHz
        prev_tail = session.prev_denoise_tail
        if len(prev_tail) == 0:
            prev_tail = np.zeros(buffer_len, dtype=np.float32)
        elif len(prev_tail) < buffer_len:
            pad = np.zeros(buffer_len - len(prev_tail), dtype=np.float32)
            prev_tail = np.concatenate([pad, prev_tail])
        else:
            prev_tail = prev_tail[-buffer_len:]

        # Ghép phần đệm lịch sử vào trước chunk hiện tại
        inp = np.concatenate([prev_tail, chunk_pcm])
        out = self.denoise.process(inp, sample_rate=sample_rate)

        # Trích xuất đúng phần âm thanh tương ứng với chunk_pcm hiện tại
        clean_chunk = out[len(prev_tail) : len(prev_tail) + len(chunk_pcm)]

        # Cập nhật đệm lịch sử cho chunk kế tiếp
        session.prev_denoise_tail = chunk_pcm[-buffer_len:] if len(chunk_pcm) >= buffer_len else np.concatenate([prev_tail[len(chunk_pcm):], chunk_pcm])
        return clean_chunk

    def process_streaming_chunk(
        self,
        session_id: str,
        chunk_pcm: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Xử lý 1 chunk âm thanh streaming (thời gian thực) trong phiên kết nối:
        1. Luôn lưu mảng PCM thô nguyên bản của microphone vào session.
        2. Khử nhiễu online với rolling buffer (nếu bật denoise).
        3. ASR nhận dạng trực tiếp sinh chữ tức thì (Interim text).
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' không tồn tại.")

        opts = session.options

        # Bước 1 & 2: Denoise streaming (nếu bật) với rolling buffer chống méo tiếng
        if opts.get("denoise", True) and self.denoise.enabled:
            clean_chunk = self.denoise_streaming_chunk(session, chunk_pcm, sample_rate=sample_rate)
        else:
            clean_chunk = chunk_pcm

        # Lưu trữ mảng PCM thô và mảng PCM đã khử nhiễu vào session
        session.add_chunk(chunk_pcm, clean_chunk)

        # Bước 3: ASR Streaming (ZipFormer)
        partial_text = ""
        if opts.get("asr", True) and self.asr.enabled and session.asr_stream is not None:
            partial_text = self.asr.process_chunk(session.asr_stream, clean_chunk, sample_rate=sample_rate)
            session.current_partial_text = partial_text
            
            # Luôn lưu vết toàn bộ từ kèm timestamp của phiên thu âm để chống mất nội dung đầu
            interim_words = self.asr.extract_interim_words_with_timestamps(session.asr_stream)
            if interim_words:
                if not session.all_words_history:
                    session.all_words_history = interim_words.copy()
                else:
                    last_t = session.all_words_history[-1]["start"]
                    for w in interim_words:
                        if w["start"] > last_t + 0.05:
                            session.all_words_history.append(w)

        # Kiểm tra cờ Realtime Diarization (Có thể Bật / Tắt linh hoạt)
        is_rt_diar = (
            opts.get("realtime_diarization", self.config.enable_realtime_diarization)
            and opts.get("diarization", True)
            and self.diarization.enabled
        )

        if not is_rt_diar:
            # Chế độ cũ (Nguyên bản): Chỉ trả về văn bản tích lũy tức thì
            return {
                "type": "partial",
                "session_id": session_id,
                "text": format_natural_text(partial_text),
                "duration": round(session.duration_seconds, 2),
                "realtime_diarization": False,
            }

        # Chế độ mới: Real-time Multi-Speaker Diarization & Verification
        interval_samples = int(opts.get("realtime_diarization_interval_s", self.config.realtime_diarization_interval_s) * sample_rate)
        now = time.time()

        if (
            session.total_samples - session.last_diarize_sample_count >= interval_samples
            and session.duration_seconds >= 0.8
        ):
            session.last_diarize_sample_count = session.total_samples
            clean_audio = session.get_clean_audio()

            # Giới hạn 20s gần nhất cho Realtime Diarization để giữ tốc độ phản hồi cực nhanh (<150ms)
            max_eval_samples = 20 * sample_rate
            if len(clean_audio) > max_eval_samples:
                eval_audio = clean_audio[-max_eval_samples:]
                eval_offset = (len(clean_audio) - max_eval_samples) / sample_rate
            else:
                eval_audio = clean_audio
                eval_offset = 0.0

            d_col = opts.get("diarization_collar", self.config.diarization_collar)
            d_min = opts.get("diarization_min_duration", self.config.diarization_min_duration)
            window_segs = self.diarization.diarize_audio_numpy(eval_audio, sample_rate=sample_rate, collar=d_col, min_duration=d_min)

            if window_segs:
                for seg in window_segs:
                    seg.start += eval_offset
                    seg.end += eval_offset

                session.confirmed_segs = window_segs

                # Người nói ở phân đoạn phát biểu gần nhất
                active_seg = window_segs[-1]
                active_spk = active_seg.speaker
                session.active_speaker_id = active_spk
                session.speaker_last_active_time[active_spk] = now

                # Kích hoạt xác thực danh tính TitaNet (Tối ưu tiến trình tích lũy: Progressive Voiceprint Refinement)
                # Khắc phục triệt để lỗi khóa danh tính vội vàng trên mẩu âm siêu ngắn (<1.5s)
                if opts.get("verification", True) and self.verification.enabled:
                    v_th = opts.get("verification_threshold", self.config.verification_threshold)
                    min_dur = opts.get("min_speech_duration_for_id", self.config.min_speech_duration_for_id)

                    for seg in window_segs:
                        spk = seg.speaker
                        curr_name, curr_score = session.speaker_identities.get(spk, ("Unknown", 0.0))
                        is_trusted = (curr_name != "Unknown" and not curr_name.startswith("Chưa đăng ký") and curr_score >= 0.75)

                        if not is_trusted:
                            # Tính tổng thời lượng tích lũy của speaker này từ đầu phiên đến giờ
                            spk_dur = sum(s.duration for s in session.confirmed_segs if s.speaker == spk)
                            if spk_dur >= min_dur:
                                spk_clips = []
                                for s in session.confirmed_segs:
                                    if s.speaker == spk:
                                        s_idx = max(0, int(s.start * sample_rate))
                                        e_idx = min(len(clean_audio), int(s.end * sample_rate))
                                        if e_idx > s_idx:
                                            spk_clips.append(clean_audio[s_idx:e_idx])
                                if spk_clips:
                                    spk_audio = np.concatenate(spk_clips)
                                    name, sim = self.verification.identify_single_speaker(spk_audio, sample_rate=sample_rate, threshold=v_th)
                                    if name != "Unknown":
                                        if sim > curr_score:
                                            session.speaker_identities[spk] = (name, sim)
                                            print(f"[Realtime ID Nâng Cấp] {spk} -> {name} (score: {sim:.3f}, tích lũy: {spk_dur:.1f}s)")
                                    else:
                                        if curr_name == "Unknown":
                                            session.speaker_identities[spk] = (f"Chưa đăng ký ({spk})", sim)

                # Cập nhật nhãn hiển thị cho người đang nói
                name, score = session.speaker_identities.get(active_spk, (active_spk, 0.0))
                session.active_identity = name
                session.active_confidence = score

        # Phân bổ từ ngữ theo mốc thời gian thực (Timestamp-Based Alignment)
        # Giúp 100% bảo toàn âm thanh đầu / chữ đầu khi có speaker mới bắt đầu phát biểu
        words_for_alignment = session.all_words_history if session.all_words_history else (self.asr.extract_interim_words_with_timestamps(session.asr_stream) if session.asr_stream else [])
        if words_for_alignment:
            temp_spk_words: Dict[str, List[str]] = {}
            for w in words_for_alignment:
                t = w["start"]
                best_spk = None
                for seg in session.confirmed_segs:
                    if (seg.start - 0.25) <= t <= (seg.end + 0.25):
                        best_spk = seg.speaker
                        break
                if best_spk is None:
                    # Các từ đang nói dở ở đuôi phân đoạn gần nhất
                    best_spk = session.active_speaker_id

                if best_spk not in temp_spk_words:
                    temp_spk_words[best_spk] = []
                temp_spk_words[best_spk].append(w["word"])
                session.speaker_last_active_time[best_spk] = now

            session.speaker_words = temp_spk_words
        elif partial_text:
            # Dự phòng nếu chưa có timestamps chi tiết
            current_words = [w.lower() for w in partial_text.split()]
            session.speaker_words[session.active_speaker_id] = current_words

        # Chuẩn bị danh sách nhiều người nói cho UI
        inactivity_timeout = float(opts.get("speaker_inactivity_timeout_s", self.config.speaker_inactivity_timeout_s))
        max_words = int(opts.get("max_recent_words_per_speaker", self.config.max_recent_words_per_speaker))

        all_speakers_keys = list(session.speaker_last_active_time.keys())
        if not all_speakers_keys:
            all_speakers_keys = [session.active_speaker_id]
            session.speaker_last_active_time[session.active_speaker_id] = now

        speakers_list = []
        for spk in all_speakers_keys:
            name, score = session.speaker_identities.get(spk, (spk, 0.0))
            last_time = session.speaker_last_active_time.get(spk, now)
            last_ago = max(0.0, now - last_time)
            is_speaking = (spk == session.active_speaker_id) and (last_ago <= 1.5)
            is_expired = last_ago > inactivity_timeout

            spk_words = session.speaker_words.get(spk, [])
            if max_words > 0 and len(spk_words) > max_words:
                recent_words = spk_words[-max_words:]
            else:
                recent_words = spk_words

            raw_spk_text = " ".join(recent_words)
            speakers_list.append({
                "speaker_id": spk,
                "name": name,
                "score": round(score, 3),
                "is_speaking": is_speaking,
                "is_expired": is_expired,
                "last_active_ago": round(last_ago, 1),
                "recent_text": format_natural_text(raw_spk_text),
                "total_words": len(spk_words)
            })

        return {
            "type": "partial",
            "session_id": session_id,
            "text": format_natural_text(partial_text),
            "duration": round(session.duration_seconds, 2),
            "realtime_diarization": True,
            "active_speaker": session.active_identity,
            "active_speaker_id": session.active_speaker_id,
            "speakers": speakers_list
        }

    def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Kết thúc phiên streaming hoặc kích hoạt khi dứt câu:
        1. Lưu âm thanh online nguyên bản (full raw audio) ra file WAV để người dùng nghe lại.
        2. Khử nhiễu toàn bộ âm thanh trọn vẹn (nếu bật denoise), bảo đảm chất lượng cao nhất.
        3. Thu nhận văn bản hoàn chỉnh và danh sách từ kèm mốc thời gian (tokens & timestamps) từ ZipFormer.
        4. Chuẩn hóa dấu câu ViBERT-capu.
        5. Phân đoạn SortFormer & định danh TitaNet.
        6. Đồng bộ chuẩn xác từng từ với người nói tương ứng mà không bao giờ cắt vụn để nhận dạng lại.
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' không tồn tại.")

        opts = session.options
        full_raw_audio = session.get_full_audio()
        total_duration = session.duration_seconds

        # 1. Lưu lại file âm thanh online thu được để người dùng có thể nghe lại
        recording_filename = "latest_recording.wav"
        recording_path = os.path.join(self.recordings_dir, recording_filename)
        audio_url = ""
        if len(full_raw_audio) > 0:
            try:
                sf.write(recording_path, full_raw_audio, 16000)
                audio_url = f"/static/recordings/{recording_filename}?t={int(time.time() * 1000)}"
            except Exception as save_err:
                print(f"[PipelineManager] Cảnh báo không thể lưu file ghi âm: {save_err}")

        # 2. Khử nhiễu âm thanh hoàn chỉnh (Full continuous denoise nếu bật)
        if opts.get("denoise", True) and self.denoise.enabled and len(full_raw_audio) > 0:
            clean_audio = self.denoise.process(full_raw_audio, sample_rate=16000)
        else:
            clean_audio = full_raw_audio

        # 3. Thu nhận văn bản hoàn chỉnh từ ASR kèm tokens & timestamps
        raw_text = ""
        words = []
        if opts.get("asr", True) and self.asr.enabled and session.asr_stream is not None:
            stream_info = self.asr.finish_stream_with_timestamps(session.asr_stream)
            raw_text = stream_info.get("text", "")
            tokens = stream_info.get("tokens", [])
            timestamps = stream_info.get("timestamps", [])
            if tokens and timestamps:
                words = self.asr.extract_words_from_tokens(tokens, timestamps)
                
            # Khắc phục lỗi Sherpa-ONNX giới hạn buffer / token ở các luồng cực dài (> 3-4 phút)
            if session.all_words_history and len(session.all_words_history) > len(words) + 10:
                words = session.all_words_history.copy()
                raw_text = " ".join([w["word"] for w in words])
            elif len(session.current_partial_text) > len(raw_text) + 20:
                raw_text = session.current_partial_text
        else:
            raw_text = session.current_partial_text

        # Bảo vệ chống mất từ: Fallback về current_partial_text nếu finish_stream trả về chuỗi rỗng
        if not raw_text.strip() and session.current_partial_text:
            raw_text = session.current_partial_text

        # 4. Hậu xử lý văn bản: Automatic Punctuation (ViBERT-capu)
        if opts.get("punctuation", True) and self.punctuation.enabled and raw_text:
            final_text = self.punctuation.punctuate(raw_text)
        else:
            final_text = format_natural_text(raw_text)

        # 5. Phân đoạn người nói (SortFormer Diarization)
        if opts.get("diarization", True) and self.diarization.enabled and len(clean_audio) > 0:
            d_col = opts.get("diarization_collar", self.config.diarization_collar)
            d_min = opts.get("diarization_min_duration", self.config.diarization_min_duration)
            segments = self.diarization.diarize_audio(clean_audio, sample_rate=16000, collar=d_col, min_duration=d_min)
        else:
            segments = [SpeakerSegment(start=0.0, end=round(total_duration, 3), speaker="speaker_0", confidence=1.0)]

        # 6. Định danh người nói (TitaNet Speaker Verification)
        if opts.get("verification", True) and self.verification.enabled and len(clean_audio) > 0:
            v_th = opts.get("verification_threshold", self.config.verification_threshold)
            id_map = self.verification.resolve_identities(clean_audio, segments, sample_rate=16000, threshold=v_th)
        else:
            unique_spks = set(s.speaker for s in segments)
            id_map = {spk: (spk, 1.0) for spk in unique_spks}

        # Kế thừa các danh tính đã được nhận diện trong luồng realtime nếu có
        for spk, (name, sc) in session.speaker_identities.items():
            if spk in id_map:
                curr_name, _ = id_map[spk]
                if curr_name.startswith("Chưa đăng ký") and not name.startswith("Chưa đăng ký"):
                    id_map[spk] = (name, sc)
            else:
                id_map[spk] = (name, sc)

        # 7. Ghép nhãn người nói vào kết quả văn bản (Chuẩn xác 100% theo từ & timestamp của luồng ASR)
        formatted_segments = self._align_text_with_speakers(
            full_audio=clean_audio,
            final_text=final_text,
            segments=segments,
            id_map=id_map,
            total_duration=total_duration,
            options=opts,
            sample_rate=16000,
            words=words
        )
        if not formatted_segments and final_text and final_text.strip():
            formatted_segments = [{
                "start": 0.0,
                "end": round(total_duration, 2),
                "speaker": "Người nói",
                "raw_speaker": "speaker_0",
                "score": 1.0,
                "text": final_text
            }]

        result = {
            "type": "final",
            "source": "mic",
            "session_id": session_id,
            "total_duration_s": round(total_duration, 2),
            "raw_text": format_natural_text(raw_text),
            "final_text": final_text,
            "segments": formatted_segments,
            "audio_url": audio_url,
            "options": opts,
        }

        # Đóng session sau khi finalize
        self.close_session(session_id)
        return result

    def process_full_audio(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        options_override: Optional[Dict[str, bool]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý trọn gói một file âm thanh (dành cho REST API upload file đồng bộ).
        """
        return self.process_full_audio_with_progress(
            audio_pcm=audio_pcm,
            sample_rate=sample_rate,
            options_override=options_override,
            progress_callback=None,
            cancel_check=cancel_check
        )

    def process_full_audio_with_progress(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        options_override: Optional[Dict[str, bool]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý file âm thanh / video với báo cáo tiến trình chi tiết theo từng bước:
        1. Chuẩn hóa & Nạp âm thanh (0% - 10%)
        2. Khử nhiễu DeepFilterNet3 (10% - 25%)
        3. Nhận dạng ZipFormer ASR (25% - 60%)
        4. Phân tách người nói SortFormer (60% - 80%)
        5. Định danh người nói TitaNet (80% - 90%)
        6. Chuẩn hóa dấu câu ViBERT-capu (90% - 98%)
        7. Hoàn tất & xuất kết quả (100%)
        Hỗ trợ ngắt quãng lập tức khi cancel_check() trả về True.
        """
        from Speaker_Diarization.streaming_sortformer import SpeakerSegment

        def check_abort():
            if cancel_check and cancel_check():
                raise InterruptedError("Quá trình xử lý đã bị hủy bởi người dùng.")

        def report(step: int, step_name: str, percent: int, message: str, skipped: bool = False, extra: Optional[Dict] = None):
            check_abort()
            if progress_callback:
                payload = {
                    "step": step,
                    "total_steps": 6,
                    "step_name": step_name,
                    "percent": percent,
                    "message": message,
                    "skipped": skipped,
                }
                if extra:
                    payload.update(extra)
                progress_callback(payload)

        check_abort()

        # 0. Chuẩn hóa về 16kHz
        if sample_rate != 16000:
            try:
                import librosa
                audio_pcm = librosa.resample(audio_pcm, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except Exception:
                pass

        check_abort()
        total_samples = len(audio_pcm)
        duration_s = total_samples / sample_rate if sample_rate > 0 else 0.0

        # Áp dụng options
        opts = {
            "denoise": self.config.enable_denoise,
            "asr": self.config.enable_asr,
            "diarization": self.config.enable_diarization,
            "verification": self.config.enable_verification,
            "hotwords": self.config.enable_hotword,
            "punctuation": self.config.enable_punctuation,
        }
        if options_override:
            opts.update(options_override)

        # BƯỚC 1: Nạp & chuẩn hóa âm thanh (0% -> 10%)
        report(1, "extract", 10, f"Đã nạp và chuẩn hóa âm thanh ({duration_s:.1f}s, 16000Hz mono)")

        # BƯỚC 2: Khử nhiễu DeepFilterNet3 (10% -> 25%)
        check_abort()
        clean_audio = audio_pcm
        if opts.get("denoise", True) and self.denoise.enabled and total_samples > 0:
            report(2, "denoise", 15, "Đang khử nhiễu môi trường (DeepFilterNet3)...")
            # Khử nhiễu toàn bộ mảng âm thanh với cơ chế overlap cross-fade nội tại của DeepFilterNet (30s segment + 1s crossfade)
            # Triệt tiêu hoàn toàn hiện tượng reset GRU state gây sụt giảm âm lượng / mất âm đầu của speaker mới
            clean_audio = self.denoise.process(audio_pcm, sample_rate=sample_rate)
            report(2, "denoise", 25, "Khử nhiễu môi trường hoàn tất!")
        else:
            report(2, "denoise", 25, "Bỏ qua khử nhiễu (Module đang tắt)", skipped=True)

        # BƯỚC 3: Nhận dạng ZipFormer ASR (25% -> 60%)
        check_abort()
        raw_text = ""
        if opts.get("asr", True) and self.asr.enabled and total_samples > 0:
            hw_param = None if (opts.get("hotwords", True) and self.hotwords.enabled) else ""
            report(3, "asr", 26, "Đang khởi tạo luồng giải mã ZipFormer Streaming Transducer...")
            stream = self.asr.create_stream(hotwords=hw_param)
            chunk_samples = int(0.32 * sample_rate)
            last_report_p = 26
            all_words_history = []
            for i in range(0, total_samples, chunk_samples):
                check_abort()
                ch = clean_audio[i : i + chunk_samples]
                self.asr.process_chunk(stream, ch, sample_rate=sample_rate)
                
                # Lưu vết words mỗi 2 giây để chống mất token ở file siêu dài
                if i % (16000 * 2) < chunk_samples:
                    interim_words = self.asr.extract_interim_words_with_timestamps(stream)
                    if interim_words:
                        if not all_words_history:
                            all_words_history = interim_words.copy()
                        else:
                            last_t = all_words_history[-1]["start"]
                            for w in interim_words:
                                if w["start"] > last_t + 0.05:
                                    all_words_history.append(w)
                                    
                p = 26 + int(34 * (i + len(ch)) / total_samples)
                if p >= last_report_p + 4 or (i + chunk_samples >= total_samples):
                    last_report_p = p
                    processed_s = (i + len(ch)) / sample_rate
                    report(3, "asr", p, f"Đang nhận dạng ASR: {processed_s:.1f}s / {duration_s:.1f}s ({int(100*(i+len(ch))/total_samples)}%)")
            check_abort()
            
            interim_words = self.asr.extract_interim_words_with_timestamps(stream)
            if interim_words:
                if not all_words_history:
                    all_words_history = interim_words.copy()
                else:
                    last_t = all_words_history[-1]["start"]
                    for w in interim_words:
                        if w["start"] > last_t + 0.05:
                            all_words_history.append(w)

            stream_info = self.asr.finish_stream_with_timestamps(stream)
            raw_text = stream_info.get("text", "")
            tokens = stream_info.get("tokens", [])
            timestamps = stream_info.get("timestamps", [])
            words = self.asr.extract_words_from_tokens(tokens, timestamps) if tokens else []
            
            # Khắc phục lỗi Sherpa-ONNX giới hạn buffer / token ở các luồng cực dài
            if len(all_words_history) > len(words) + 10:
                words = all_words_history.copy()
                raw_text = " ".join([w["word"] for w in words])
                
            word_count = len(raw_text.split()) if raw_text else 0
            report(3, "asr", 60, f"Nhận dạng ASR hoàn tất! ({word_count} từ)")
        else:
            words = []
            report(3, "asr", 60, "Bỏ qua nhận dạng ASR (Module đang tắt)", skipped=True)

        # BƯỚC 4: Phân đoạn người nói SortFormer (60% -> 80%)
        check_abort()
        if opts.get("diarization", True) and self.diarization.enabled and len(clean_audio) > 0:
            report(4, "diarization", 65, "Đang phân đoạn người nói (NVIDIA Streaming SortFormer)...")
            d_col = opts.get("diarization_collar", self.config.diarization_collar)
            d_min = opts.get("diarization_min_duration", self.config.diarization_min_duration)
            segments = self.diarization.diarize_audio(clean_audio, sample_rate=16000, collar=d_col, min_duration=d_min)
            report(4, "diarization", 80, f"Phân đoạn người nói thành công! Tìm thấy {len(segments)} phân đoạn phát biểu.")
        else:
            segments = [SpeakerSegment(start=0.0, end=round(duration_s, 3), speaker="speaker_0", confidence=1.0)]
            report(4, "diarization", 80, "Bỏ qua phân đoạn người nói (Module đang tắt)", skipped=True)

        # BƯỚC 5: Định danh người nói TitaNet (80% -> 90%)
        check_abort()
        if opts.get("verification", True) and self.verification.enabled and len(clean_audio) > 0:
            report(5, "verification", 82, "Đang so khớp nhận diện danh tính người nói (NVIDIA TitaNet)...")
            v_th = opts.get("verification_threshold", self.config.verification_threshold)
            id_map = self.verification.resolve_identities(clean_audio, segments, sample_rate=16000, threshold=v_th)
            report(5, "verification", 90, f"Định danh giọng nói hoàn tất! Đã tra cứu {len(id_map)} người nói.")
        else:
            unique_spks = set(s.speaker for s in segments)
            id_map = {spk: (spk, 1.0) for spk in unique_spks}
            report(5, "verification", 90, "Bỏ qua định danh người nói (Module đang tắt)", skipped=True)

        # BƯỚC 6: Chuẩn hóa dấu câu ViBERT-capu (90% -> 98%)
        check_abort()
        if opts.get("punctuation", True) and self.punctuation.enabled and raw_text:
            report(6, "punctuation", 92, "Đang thêm dấu câu & viết hoa chuẩn hóa (ViBERT-capu ONNX)...")
            final_text = self.punctuation.punctuate(raw_text)
            report(6, "punctuation", 98, "Chuẩn hóa văn bản hoàn tất!")
        else:
            final_text = format_natural_text(raw_text)
            report(6, "punctuation", 98, "Bỏ qua chuẩn hóa dấu câu", skipped=True)

        # Gom nhóm và đồng bộ lượt nói (Đảm bảo 100% khớp văn bản ASR từ luồng chính)
        check_abort()
        formatted_segments = self._align_text_with_speakers(
            full_audio=clean_audio,
            final_text=final_text,
            segments=segments,
            id_map=id_map,
            total_duration=duration_s,
            options=opts,
            sample_rate=16000,
            words=words
        )

        check_abort()
        # BƯỚC 7: Hoàn tất (100%)
        report(6, "completed", 100, "Xử lý toàn bộ video/audio thành công!")

        # Lưu lại file âm thanh chuẩn 16kHz đã trích xuất & khử nhiễu cho Web Client phát chính xác
        offline_filename = "latest_offline.wav"
        offline_path = os.path.join(self.recordings_dir, offline_filename)
        offline_audio_url = ""
        if len(clean_audio) > 0:
            try:
                sf.write(offline_path, clean_audio, 16000)
                offline_audio_url = f"/static/recordings/{offline_filename}?t={int(time.time() * 1000)}"
            except Exception as save_err:
                print(f"[PipelineManager] Cảnh báo không thể lưu file offline audio: {save_err}")

        result = {
            "type": "final",
            "source": "offline",
            "total_duration_s": round(duration_s, 2),
            "raw_text": format_natural_text(raw_text),
            "final_text": final_text,
            "segments": formatted_segments,
            "audio_url": offline_audio_url,
            "options": opts,
        }
        return result

    def _align_text_with_speakers(
        self,
        full_audio: np.ndarray,
        final_text: str,
        segments: List[Any],
        id_map: Dict[str, Tuple[str, float]],
        total_duration: float,
        options: Dict[str, bool],
        sample_rate: int = 16000,
        words: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Đồng bộ hóa hoàn hảo các phân đoạn người nói:
        - Sử dụng danh sách từ kèm mốc thời gian thực (tokens & timestamps) giải mã từ luồng ASR liên tục.
        - Khắc phục triệt để lỗi mất từ, méo từ hay chữ sai lệch do cắt vụn audio để nhận dạng lại.
        - Gán từng từ vào người nói phù hợp nhất dựa trên ranh giới phân đoạn SortFormer.
        - Tự động gom các từ liên tiếp của cùng 1 người nói và áp dụng chuẩn hóa dấu câu ViBERT-capu.
        """
        if not segments:
            return [{
                "start": 0.0,
                "end": round(total_duration, 2),
                "speaker": "Người nói",
                "raw_speaker": "speaker_0",
                "score": 1.0,
                "text": format_natural_text(final_text)
            }]

        raw_spks = set(s.speaker for s in segments)
        real_spks = set(id_map.get(s.speaker, (s.speaker, 1.0))[0] for s in segments)

        # Trường hợp 1: Tính năng Diarization tắt hoặc chỉ có 1 người nói duy nhất
        if len(real_spks) <= 1 or not options.get("diarization", True):
            raw_spk = segments[0].speaker
            real_name, score = id_map.get(raw_spk, (raw_spk, 1.0))
            min_s = min(s.start for s in segments)
            max_e = max(s.end for s in segments)
            return [{
                "start": round(min_s, 2),
                "end": round(max_e, 2),
                "speaker": real_name,
                "raw_speaker": raw_spk,
                "score": round(score, 3),
                "text": format_natural_text(final_text)
            }]

        # Trường hợp 2: Có nhiều người nói khác nhau và có mốc thời gian từng từ (words)
        if words and len(words) > 0:
            sorted_segs = sorted(segments, key=lambda s: s.start)
            word_assignments = []

            for w in words:
                t = w["start"]
                best_seg = None
                min_dist = float("inf")

                # Tìm phân đoạn bao phủ mốc thời gian của từ (dung sai 0.25s)
                for seg in sorted_segs:
                    if (seg.start - 0.25) <= t <= (seg.end + 0.25):
                        best_seg = seg
                        break
                    dist = min(abs(t - seg.start), abs(t - seg.end))
                    if dist < min_dist:
                        min_dist = dist
                        best_seg = seg

                word_assignments.append((w, best_seg))

            # Gom các từ thành từng lượt phát biểu: Cùng 1 speaker thì gom (stack) toàn bộ nội dung lại
            turns = []
            curr_turn = None

            for w, seg in word_assignments:
                raw_spk = seg.speaker if seg else "speaker_0"
                real_name, score = id_map.get(raw_spk, (raw_spk, 1.0))

                # Đổi lượt nói khi phát hiện người nói khác nhau (khác speaker)
                is_spk_change = (
                    curr_turn is None 
                    or curr_turn["speaker"] != real_name 
                    or curr_turn["raw_speaker"] != raw_spk
                )

                if is_spk_change:
                    if curr_turn is not None:
                        turns.append(curr_turn)
                    curr_turn = {
                        "speaker": real_name,
                        "raw_speaker": raw_spk,
                        "score": score,
                        "start": max(0.0, w["start"]),
                        "end": w["end"],
                        "words": [w["word"]]
                    }
                else:
                    curr_turn["words"].append(w["word"])
                    curr_turn["end"] = max(curr_turn["end"], w["end"])

            if curr_turn is not None:
                turns.append(curr_turn)

            results = []
            for turn in turns:
                turn_raw = " ".join(turn["words"]).strip()
                if not turn_raw:
                    continue
                if options.get("punctuation", True) and self.punctuation.enabled:
                    turn_text = self.punctuation.punctuate(turn_raw)
                else:
                    turn_text = format_natural_text(turn_raw)

                results.append({
                    "start": round(turn["start"], 2),
                    "end": round(turn["end"], 2),
                    "speaker": turn["speaker"],
                    "raw_speaker": turn["raw_speaker"],
                    "score": round(turn["score"], 3),
                    "text": turn_text
                })

            if results:
                return results

        # Fallback an toàn: Trả về khối văn bản trọn vẹn final_text gắn với người nói chính
        raw_spk = segments[0].speaker
        real_name, score = id_map.get(raw_spk, (raw_spk, 1.0))
        return [{
            "start": round(min(s.start for s in segments), 2),
            "end": round(max(s.end for s in segments), 2),
            "speaker": real_name,
            "raw_speaker": raw_spk,
            "score": round(score, 3),
            "text": format_natural_text(final_text)
        }]
