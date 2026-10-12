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
    from .modules.asr_module import ASRModule, OfflineASRModule
    from .modules.diarization_module import DiarizationModule
    from .modules.verification_module import VerificationModule
    from .modules.hotword_module import HotwordModule
    from .modules.punctuation_module import PunctuationModule
    from .modules.summary_module import SummaryModule
except ImportError:
    from config import PipelineConfig
    from modules.denoise_module import DenoiseModule
    from modules.asr_module import ASRModule, OfflineASRModule
    from modules.diarization_module import DiarizationModule
    from modules.verification_module import VerificationModule
    from modules.hotword_module import HotwordModule
    from modules.punctuation_module import PunctuationModule
    from modules.summary_module import SummaryModule

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


RAW_HALLUCINATED_PATTERNS = [
    r'\bkhông biết đăng ký kênh\b',
    r'\bhãy đăng ký kênh\b',
    r'\bnhớ đăng ký kênh\b',
    r'\bđừng quên đăng ký kênh\b',
    r'\bđăng ký kênh ủng hộ\b',
    r'\bđăng ký kênh\b',
    r'\bđăng kí kênh\b',
    r'\bxin chào và hẹn gặp lại\b',
    r'\bhẹn gặp lại\b',
    r'\bnhớ bấm like và subscribe\b',
    r'\bnhớ like và subscribe\b',
    r'\blike và subscribe\b',
    r'\bnhấn chuông thông báo\b',
    r'\bnhấn chuông\b',
    r'\bem đây bất ngờ\b',
    r'\bcảm ơn các bạn đã theo dõi\b',
    r'\bchúc các bạn xem video vui vẻ\b',
]
HALLUCINATED_PATTERNS = sorted(RAW_HALLUCINATED_PATTERNS, key=len, reverse=True)


def clean_asr_hallucinations(text: str) -> str:
    """
    Loại bỏ các cụm từ ảo giác (YouTube bias) sinh ra do Zipformer giải mã trên nền nhạc/im lặng:
    - 'đăng ký kênh', 'hẹn gặp lại', 'không biết đăng ký kênh', 'em đây bất ngờ', v.v.
    """
    if not text:
        return ""
    cleaned = text
    # Khử từ ảo giác sinh ra trên nhạc dạo đầu ngay trước lời chào
    cleaned = re.sub(r'^\s*không biết\s+(?=hello|xin chào)', '', cleaned, flags=re.IGNORECASE)
    # Khử lỗi nuốt âm đầu biến 'kính đại biểu' thành 'ký đại biểu'
    cleaned = re.sub(r'\bký đại biểu\b', 'Kính đại biểu', cleaned, flags=re.IGNORECASE)
    for pat in HALLUCINATED_PATTERNS:
        cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+([,.:;?!])', r'\1', cleaned)
    cleaned = re.sub(r'([,.:;?!])\s*([,.:;?!])+', r'\1', cleaned)
    cleaned = re.sub(r'^\s*[,.:;?!]+\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def deduplicate_consecutive_phrases(text: str) -> str:
    """
    Khử các cụm từ bị lặp lại liên tiếp do hiện tượng chốt lát cắt ASR (boundary overlap).
    Ví dụ: 'hôm nay hôm nay hôm nay ngày mười ba' -> 'hôm nay ngày mười ba'
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= 1:
        return text

    i = 0
    new_words = []
    while i < len(words):
        matched = False
        for n in range(4, 0, -1):
            if i + 2 * n <= len(words):
                ngram1 = [re.sub(r'[,.:;?!]', '', w.lower()) for w in words[i : i + n]]
                ngram2 = [re.sub(r'[,.:;?!]', '', w.lower()) for w in words[i + n : i + 2 * n]]
                if ngram1 == ngram2:
                    new_words.extend(words[i : i + n])
                    i += n
                    while i + n <= len(words):
                        next_ngram = [re.sub(r'[,.:;?!]', '', w.lower()) for w in words[i : i + n]]
                        if next_ngram == ngram1:
                            i += n
                        else:
                            break
                    matched = True
                    break
        if not matched:
            new_words.append(words[i])
            i += 1
    return ' '.join(new_words)


def merge_transcripts_without_overlap(prev_text: str, next_text: str) -> str:
    """
    Nối hai đoạn văn bản ASR mà không lặp lại các từ ở ranh giới giao nhau.
    Ví dụ: prev='ngày hôm nay', next='hôm nay trời đẹp' -> 'ngày hôm nay trời đẹp'
    """
    if not prev_text.strip():
        return next_text.strip()
    if not next_text.strip():
        return prev_text.strip()

    prev_words = prev_text.strip().split()
    next_words = next_text.strip().split()

    max_k = min(6, len(prev_words), len(next_words))
    best_overlap = 0

    for k in range(max_k, 0, -1):
        suffix = [re.sub(r'[,.:;?!]', '', w.lower()) for w in prev_words[-k:]]
        prefix = [re.sub(r'[,.:;?!]', '', w.lower()) for w in next_words[:k]]
        if suffix == prefix:
            best_overlap = k
            break

    if best_overlap > 0:
        merged_words = prev_words + next_words[best_overlap:]
    else:
        merged_words = prev_words + next_words

    merged = " ".join(merged_words)
    return deduplicate_consecutive_phrases(merged)


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
        self.speaker_last_active_audio_time: Dict[str, float] = {}  # {spk_id: audio_seconds}
        self.speaker_words: Dict[str, List[str]] = {}               # {spk_id: [words]}
        self.active_speaker_id: str = "speaker_0"
        self.active_identity: str = "speaker_0"
        self.active_confidence: float = 1.0
        self.last_diarize_sample_count: int = 0
        self.last_processed_word_count: int = 0
        self.confirmed_segs: List[Any] = []
        self.all_confirmed_segs: List[Any] = []

        # Trạng thái 2-Pass Streaming ASR cuốn chiếu (Nemotron VAD + ZipFormer-150M)
        self.committed_turns: List[Dict[str, Any]] = []
        self.last_committed_sample: int = 0
        self.current_open_turn_id: Optional[str] = None
        self.current_open_speaker: Optional[str] = None
        self.current_open_raw_speaker: Optional[str] = None
        self.last_active_speaker_id: Optional[str] = None

        # Trạng thái Dynamic In-Session Voiceprint Tracking (TitaNet 192-dim)
        self.speaker_embeddings: Dict[str, np.ndarray] = {}

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

        # 3.1 Khởi tạo Offline ASR Module cho chế độ Upload/Offline (Zipformer-150M)
        self.offline_asr = OfflineASRModule(
            model_dir=os.path.join(app_dir, "zipformer_offline_vi"),
            num_threads=self.config.asr_threads,
            provider="cpu",
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

        # 7. Khởi tạo Summary Module (Qwen2.5-1.5B-Instruct)
        self.summary = SummaryModule(
            model_name=self.config.summary_model,
            device=self.config.device,
            enabled=self.config.enable_summary,
            max_tokens=self.config.summary_max_tokens,
            temperature=self.config.summary_temperature,
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

        # 2. Diarization (SortFormer / Nemotron-3)
        d_col = params.get("diarization_collar")
        d_min = params.get("diarization_min_duration")
        d_chunk = params.get("diarization_chunk_len")
        d_model = params.get("diarization_model")
        
        if d_model and d_model != self.diarization.model_name:
            print(f"[PipelineManager] Phát hiện thay đổi mô hình Diarization sang {d_model}. Đang khởi tạo lại...")
            self.diarization = DiarizationModule(
                model_name=d_model,
                device=self.config.device,
                chunk_len=d_chunk if d_chunk else self.config.diarization_chunk_len,
                collar=d_col if d_col else self.config.diarization_collar,
                min_duration=d_min if d_min else self.config.diarization_min_duration,
                enabled=self.config.enable_diarization,
            )
        else:
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

        # 5. Summary (Qwen2.5-1.5B)
        s_tokens = params.get("summary_max_tokens")
        s_temp = params.get("summary_temperature")
        self.summary.set_parameters(max_tokens=s_tokens, temperature=s_temp)
        if "enable_summary" in params:
            self.summary.set_enabled(bool(params.get("enable_summary")))

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
                    "title": "Phân Đoạn Người Nói (Diarization)",
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
                            "key": "diarization_model",
                            "label": "Mô Hình Phân Đoạn (Diarization Model)",
                            "type": "select",
                            "options": [
                                {"value": "nvidia/Nemotron-3-Diarization", "label": "Nemotron-3 (Chính xác cao, nhiều speakers)"},
                                {"value": "nvidia/diar_streaming_sortformer_4spk-v2.1", "label": "SortFormer (Tối đa 4 speakers, nhanh gọn)"}
                            ],
                            "default": "nvidia/Nemotron-3-Diarization",
                            "description": "Lựa chọn mô hình cho Diarization."
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
                        },
                        {
                            "key": "enable_2pass_streaming",
                            "label": "Bật Chốt Lượt Cuốn Chiếu (2-Pass ZipFormer-150M)",
                            "type": "toggle",
                            "default": True,
                            "description": "Tự động chốt từng lượt phát biểu bằng ZipFormer-150M khi người nói tạm dừng, nối tăng dần chữ theo thời gian thực."
                        },
                        {
                            "key": "turn_silence_timeout_s",
                            "label": "Ngưỡng Im Lặng Chốt Lượt (Silence Timeout)",
                            "type": "slider",
                            "min": 0.4,
                            "max": 2.0,
                            "step": 0.1,
                            "default": 0.7,
                            "unit": "s",
                            "description": "Khoảng thời gian im lặng tối thiểu để hệ thống xác nhận người nói đã tạm dừng ý và kích hoạt chốt ZipFormer-150M."
                        }
                    ]
                },
                {
                    "id": "summary",
                    "title": "Tóm Tắt Nội Dung Tự Động (Qwen2.5-1.5B-Instruct)",
                    "icon": "fa-wand-magic-sparkles",
                    "params": [
                        {
                            "key": "enable_summary",
                            "label": "Bật Tóm Tắt Tự Động",
                            "type": "toggle",
                            "default": True,
                            "description": "Tự động phân tích và tóm tắt ý chính, quyết định và phát biểu của từng người khi kết thúc ghi âm hoặc xử lý file."
                        },
                        {
                            "key": "summary_max_tokens",
                            "label": "Độ Dài Tối Đa Bản Tóm Tắt (Max Tokens)",
                            "type": "number",
                            "min": 128,
                            "max": 1024,
                            "step": 64,
                            "default": 768,
                            "unit": "tokens",
                            "description": "Giới hạn số token tối đa mà mô hình Qwen2.5 sẽ sinh ra cho bản biên bản cuộc họp."
                        },
                        {
                            "key": "summary_temperature",
                            "label": "Độ Sáng Tạo (Temperature)",
                            "type": "slider",
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.05,
                            "default": 0.3,
                            "unit": "",
                            "description": "Nhiệt độ giải mã: Thấp (0.1 - 0.3) giúp tóm tắt chính xác, trung thành với nội dung; Cao hơn giúp hành văn phong phú hơn."
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
            "summary": self.config.enable_summary,
            "realtime_diarization": self.config.enable_realtime_diarization,
            "realtime_diarization_interval_s": self.config.realtime_diarization_interval_s,
            "max_recent_words_per_speaker": self.config.max_recent_words_per_speaker,
            "speaker_inactivity_timeout_s": self.config.speaker_inactivity_timeout_s,
            "enable_2pass_streaming": self.config.enable_2pass_streaming,
            "turn_silence_timeout_s": self.config.turn_silence_timeout_s,
            "turn_min_speech_s": self.config.turn_min_speech_s,
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

    def _get_speaker_display_identity(self, session: StreamingSession, spk_id: str) -> Tuple[str, float]:
        """Trả về (display_name, confidence_score) phân biệt rõ ràng cho từng người nói."""
        if spk_id in session.speaker_identities:
            name, score = session.speaker_identities[spk_id]
            if name and name != "Unknown" and name != "Chưa đăng ký":
                return name, score

        # Tạo nhãn phân biệt mặc định theo mã speaker (ví dụ: Người nói 1 (Chưa đăng ký))
        spk_num = spk_id.replace("speaker_", "")
        display_label = f"Người nói {int(spk_num) + 1 if spk_num.isdigit() else spk_id} (Chưa đăng ký)"
        return display_label, 0.0

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

        # Chuẩn hóa về 16kHz nếu client/test truyền sample rate khác 16kHz
        if sample_rate != 16000 and len(chunk_pcm) > 0:
            try:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(sample_rate), 16000)
                chunk_pcm = resample_poly(chunk_pcm, 16000 // g, int(sample_rate) // g).astype(np.float32)
                sample_rate = 16000
            except Exception:
                pass

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

        # Chế độ mới: Real-time Multi-Speaker Diarization & Verification & 2-Pass ZipFormer-150M Commits
        interval_samples = int(opts.get("realtime_diarization_interval_s", self.config.realtime_diarization_interval_s) * sample_rate)
        now = time.time()
        turn_event = None

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
            silence_timeout_s = float(opts.get("turn_silence_timeout_s", self.config.turn_silence_timeout_s))

            # Phát hiện hoạt động giọng nói (VAD) và phân đoạn
            vad_res = self.diarization.detect_vad_activity(
                eval_audio,
                sample_rate=sample_rate,
                silence_threshold_s=silence_timeout_s
            )
            window_segs = vad_res.get("segments", [])
            is_speaking = vad_res.get("is_speaking", True)
            silence_duration_s = vad_res.get("silence_duration_s", 0.0)
            last_speech_end = vad_res.get("last_speech_end", 0.0) + eval_offset

            speaker_changed = False
            change_transition_sample = None
            prev_speaker_for_commit = None

            if window_segs:
                for seg in window_segs:
                    seg.start += eval_offset
                    seg.end += eval_offset

                # -------------------------------------------------------------
                # DYNAMIC IN-SESSION VOICEPRINT TRACKING (NVIDIA TitaNet 192-dim)
                # Khóa danh tính người nói liên tục, chống triệt để lỗi reset nhầm speaker
                # khi người nói trước trôi ra ngoài cửa sổ đánh giá cục bộ 20s
                # -------------------------------------------------------------
                if self.verification.enabled:
                    track_th = opts.get("speaker_tracking_threshold", self.config.speaker_tracking_threshold)
                    local_spk_clips = {}
                    for seg in window_segs:
                        s_idx = max(0, int((seg.start - eval_offset) * sample_rate))
                        e_idx = min(len(eval_audio), int((seg.end - eval_offset) * sample_rate))
                        if e_idx > s_idx:
                            clip = eval_audio[s_idx:e_idx]
                            if seg.speaker not in local_spk_clips:
                                local_spk_clips[seg.speaker] = []
                            local_spk_clips[seg.speaker].append(clip)

                    local_to_global_map = {}
                    for loc_spk, clips in local_spk_clips.items():
                        full_clip = np.concatenate(clips, axis=0) if clips else None
                        if full_clip is not None and len(full_clip) >= int(0.6 * sample_rate):
                            cur_emb = self.verification.extract_embedding(full_clip, sample_rate=sample_rate)
                            if cur_emb is not None:
                                best_g = None
                                best_sim = -1.0
                                for g_spk, g_emb in session.speaker_embeddings.items():
                                    sim = float(np.dot(cur_emb, g_emb))
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_g = g_spk

                                if best_g is not None and best_sim >= track_th:
                                    local_to_global_map[loc_spk] = best_g
                                    # Cập nhật centroid embedding làm mịn dần theo thời gian
                                    updated_emb = 0.8 * session.speaker_embeddings[best_g] + 0.2 * cur_emb
                                    norm = np.linalg.norm(updated_emb)
                                    if norm > 1e-6:
                                        session.speaker_embeddings[best_g] = updated_emb / norm
                                elif best_g is not None and best_sim >= 0.42:
                                    # Gần ngưỡng (0.42 <= sim < track_th): gán vào best_g để chống bùng nổ người nói ảo
                                    local_to_global_map[loc_spk] = best_g
                                    updated_emb = 0.85 * session.speaker_embeddings[best_g] + 0.15 * cur_emb
                                    norm = np.linalg.norm(updated_emb)
                                    if norm > 1e-6:
                                        session.speaker_embeddings[best_g] = updated_emb / norm
                                elif len(session.speaker_embeddings) < 8 and len(full_clip) >= int(0.8 * sample_rate):
                                    new_g = f"speaker_{len(session.speaker_embeddings)}"
                                    session.speaker_embeddings[new_g] = cur_emb
                                    local_to_global_map[loc_spk] = new_g
                                    print(f"[Speaker Tracker] Đăng ký người nói mới trong phiên: {new_g} (best_sim: {best_sim:.3f})")
                                elif best_g is not None:
                                    local_to_global_map[loc_spk] = best_g
                                else:
                                    new_g = f"speaker_{len(session.speaker_embeddings)}"
                                    session.speaker_embeddings[new_g] = cur_emb
                                    local_to_global_map[loc_spk] = new_g

                    if local_to_global_map:
                        for seg in window_segs:
                            if seg.speaker in local_to_global_map:
                                seg.speaker = local_to_global_map[seg.speaker]

                session.confirmed_segs = window_segs
                if not hasattr(session, "all_confirmed_segs") or session.all_confirmed_segs is None:
                    session.all_confirmed_segs = []
                past_segs = [s for s in session.all_confirmed_segs if s.end <= eval_offset + 0.05]
                session.all_confirmed_segs = past_segs + list(window_segs)

                # Người nói ở phân đoạn phát biểu gần nhất
                active_seg = window_segs[-1]
                active_spk = active_seg.speaker

                # Phát hiện thay đổi người nói giữa các lượt phát biểu
                prev_spk = session.last_active_speaker_id
                if prev_spk is None:
                    session.last_active_speaker_id = active_spk
                elif prev_spk != active_spk:
                    speaker_changed = True
                    prev_speaker_for_commit = prev_spk

                    # Xác định điểm chuyển giao: mốc bắt đầu của phân đoạn speaker mới sau điểm đã chốt
                    active_starts = [
                        seg.start for seg in window_segs
                        if seg.speaker == active_spk and (seg.start * sample_rate) >= session.last_committed_sample
                    ]
                    if active_starts:
                        change_transition_sample = int(active_starts[0] * sample_rate)
                    else:
                        prev_ends = [seg.end for seg in window_segs if seg.speaker == prev_spk]
                        if prev_ends:
                            change_transition_sample = int(prev_ends[-1] * sample_rate)
                        else:
                            change_transition_sample = len(clean_audio)

                    change_transition_sample = max(session.last_committed_sample, min(len(clean_audio), change_transition_sample))

                session.active_speaker_id = active_spk
                session.speaker_last_active_time[active_spk] = now
                if not hasattr(session, "speaker_last_active_audio_time"):
                    session.speaker_last_active_audio_time = {}
                session.speaker_last_active_audio_time[active_spk] = session.duration_seconds

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
                            all_segs = session.all_confirmed_segs if getattr(session, "all_confirmed_segs", None) else session.confirmed_segs
                            spk_dur = sum(s.duration for s in all_segs if s.speaker == spk)
                            if spk_dur >= min_dur:
                                spk_clips = []
                                for s in all_segs:
                                    if s.speaker == spk:
                                        s_idx = max(0, int(s.start * sample_rate))
                                        e_idx = min(len(clean_audio), int(s.end * sample_rate))
                                        if e_idx > s_idx:
                                            spk_clips.append(clean_audio[s_idx:e_idx])
                                if spk_clips:
                                    spk_audio = np.concatenate(spk_clips)
                                    name, sim = self.verification.identify_single_speaker(spk_audio, sample_rate=sample_rate, threshold=v_th)
                                    if name != "Unknown" and name != "Chưa đăng ký":
                                        if sim > curr_score:
                                            session.speaker_identities[spk] = (name, sim)
                                            print(f"[Realtime ID Nâng Cấp] {spk} -> {name} (score: {sim:.3f}, tích lũy: {spk_dur:.1f}s)")
                                    else:
                                        if curr_name == "Unknown" or curr_name == "Chưa đăng ký":
                                            spk_num = spk.replace("speaker_", "")
                                            display_label = f"Người nói {int(spk_num) + 1 if spk_num.isdigit() else spk} (Chưa đăng ký)"
                                            session.speaker_identities[spk] = (display_label, sim)

                # Cập nhật nhãn hiển thị cho người đang nói
                name, score = self._get_speaker_display_identity(session, active_spk)
                session.active_identity = name
                session.active_confidence = score

            # -----------------------------------------------------------------
            # 2-Pass Incremental ASR: Chốt cuốn chiếu bằng ZipFormer-150M
            # -----------------------------------------------------------------
            enable_2pass = opts.get("enable_2pass_streaming", self.config.enable_2pass_streaming)
            min_speech_s = float(opts.get("turn_min_speech_s", self.config.turn_min_speech_s))

            if enable_2pass and self.offline_asr.enabled and len(clean_audio) > session.last_committed_sample:
                uncommitted_samples = len(clean_audio) - session.last_committed_sample
                uncommitted_s = uncommitted_samples / sample_rate

                should_commit = False
                force_close_turn = False
                commit_spk_id = session.active_speaker_id
                cut_end_sample = len(clean_audio)

                # Trường hợp 1: Phát hiện đổi người nói (Speaker Change) -> chốt ngay lời của người nói trước
                if speaker_changed and prev_speaker_for_commit and change_transition_sample is not None:
                    prev_uncommitted_samples = change_transition_sample - session.last_committed_sample
                    if prev_uncommitted_samples >= int(min_speech_s * sample_rate):
                        should_commit = True
                        force_close_turn = True
                        commit_spk_id = prev_speaker_for_commit
                        cut_end_sample = change_transition_sample
                    else:
                        session.last_committed_sample = max(session.last_committed_sample, change_transition_sample)
                        session.current_open_turn_id = None
                        session.current_open_speaker = None
                        session.current_open_raw_speaker = None
                    session.last_active_speaker_id = session.active_speaker_id

                # Trường hợp 2: Người nói ngắt quãng (im lặng >= silence_timeout_s)
                elif not is_speaking and silence_duration_s >= silence_timeout_s:
                    speech_end_sample = int(last_speech_end * sample_rate) + int(0.15 * sample_rate)
                    # Chỉ commit khi còn âm thanh giọng nói mới chưa chốt
                    if speech_end_sample > session.last_committed_sample and (speech_end_sample - session.last_committed_sample) >= int(min_speech_s * sample_rate):
                        should_commit = True
                        # Chỉ đóng hẳn lượt khi khoảng im lặng dài (>= 3.5s)
                        # Nếu chỉ ngắt nghỉ lấy hơi (< 3.5s), giữ mở turn để gộp các câu liên tiếp của cùng người nói
                        force_close_turn = (silence_duration_s >= 3.5)
                        cut_end_sample = min(len(clean_audio), speech_end_sample)
                        commit_spk_id = session.active_speaker_id
                    else:
                        if silence_duration_s >= 3.5:
                            session.current_open_turn_id = None
                            session.current_open_speaker = None
                            session.current_open_raw_speaker = None

                # Trường hợp 3: Đang nói liên tục dài (>= 8.0s) -> commit định kỳ để tránh dồn ứ
                elif is_speaking and uncommitted_s >= 8.0:
                    # Tìm thung lũng năng lượng thấp nhất trong 0.6s gần nhất để cắt mượt mà giữa các từ
                    search_window = int(0.6 * sample_rate)
                    if len(clean_audio) > search_window:
                        recent_tail = clean_audio[-search_window:]
                        step = int(0.05 * sample_rate)
                        min_e = float('inf')
                        best_offset = search_window
                        for s_idx in range(0, len(recent_tail) - step, step):
                            e = np.sum(np.abs(recent_tail[s_idx : s_idx + step]))
                            if e < min_e:
                                min_e = e
                                best_offset = search_window - (s_idx + step // 2)
                        cut_end_sample = len(clean_audio) - best_offset
                    else:
                        cut_end_sample = len(clean_audio)
                    should_commit = True
                    force_close_turn = False
                    commit_spk_id = session.active_speaker_id

                if should_commit and (cut_end_sample - session.last_committed_sample) >= int(min_speech_s * sample_rate):
                    slice_pcm = clean_audio[session.last_committed_sample : cut_end_sample]
                    slice_start_s = session.last_committed_sample / sample_rate
                    slice_end_s = cut_end_sample / sample_rate
                    session.last_committed_sample = cut_end_sample

                    slice_rms = float(np.sqrt(np.mean(slice_pcm ** 2))) if len(slice_pcm) > 0 else 0.0
                    if slice_rms < 0.003:
                        slice_raw_text = ""
                    else:
                        # Thêm 150ms zero-padding vào đầu slice để tránh bị nuốt phụ âm đầu
                        pad_silence = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
                        slice_pcm_padded = np.concatenate([pad_silence, slice_pcm])
                        try:
                            slice_res = self.offline_asr.process_full_audio(slice_pcm_padded, sample_rate=sample_rate)
                            slice_raw_text = (slice_res.get("text") or "").strip()
                            slice_raw_text = clean_asr_hallucinations(slice_raw_text)
                        except Exception as asr_err:
                            print(f"[PipelineManager 2-Pass] Lỗi giải mã ZipFormer-150M: {asr_err}")
                            slice_raw_text = ""

                    if slice_raw_text:
                        spk_id = commit_spk_id
                        display_name, score = self._get_speaker_display_identity(session, spk_id)

                        # Kiểm tra xem có cùng người nói để nối (merge) vào lượt trước hay không
                        has_recent_turn = len(session.committed_turns) > 0
                        last_turn = session.committed_turns[-1] if has_recent_turn else None
                        collar_s = float(opts.get("diarization_collar", self.config.diarization_collar))

                        is_same_speaker = False
                        if has_recent_turn and last_turn is not None:
                            is_same_speaker_id = (
                                last_turn.get("raw_speaker") == spk_id
                                or last_turn.get("speaker") == display_name
                            )
                            if is_same_speaker_id:
                                # Khoảng cách thời gian giữa cuối lượt trước và đầu lượt mới
                                time_gap = max(0.0, slice_start_s - float(last_turn.get("end", slice_start_s)))
                                # Tự động nối nếu:
                                # 1. Turn đang mở (session.current_open_turn_id is not None)
                                # 2. HOẶC các lượt âm thanh nối tiếp nhau liền kề / ngắt nghỉ ngắn (time_gap <= max(collar_s, 0.5))
                                if session.current_open_turn_id is not None or time_gap <= max(collar_s, 0.5):
                                    is_same_speaker = True
                                    session.current_open_turn_id = last_turn["turn_id"]
                                    session.current_open_raw_speaker = spk_id
                                    session.current_open_speaker = display_name

                        if is_same_speaker:
                            curr_turn = session.committed_turns[-1]
                            curr_turn["raw_text"] = merge_transcripts_without_overlap(curr_turn["raw_text"], slice_raw_text)
                            curr_turn["text"] = format_natural_text(curr_turn["raw_text"])
                            curr_turn["end"] = round(slice_end_s, 2)
                            curr_turn["speaker"] = display_name
                            curr_turn["score"] = round(score, 3)
                            turn_event = {
                                "action": "turn_update",
                                "turn_id": curr_turn["turn_id"],
                                "speaker": curr_turn["speaker"],
                                "raw_speaker": curr_turn["raw_speaker"],
                                "score": curr_turn["score"],
                                "text": curr_turn["text"],
                                "start": curr_turn["start"],
                                "end": curr_turn["end"],
                            }
                        else:
                            new_turn_id = f"turn_{int(time.time() * 1000) % 1000000}_{uuid.uuid4().hex[:4]}"
                            new_turn = {
                                "turn_id": new_turn_id,
                                "speaker": display_name,
                                "raw_speaker": spk_id,
                                "score": round(score, 3),
                                "raw_text": slice_raw_text,
                                "text": format_natural_text(slice_raw_text),
                                "start": round(slice_start_s, 2),
                                "end": round(slice_end_s, 2),
                            }
                            session.committed_turns.append(new_turn)
                            session.current_open_turn_id = new_turn_id
                            session.current_open_speaker = display_name
                            session.current_open_raw_speaker = spk_id
                            turn_event = {
                                "action": "turn_new",
                                **new_turn
                            }

                        if force_close_turn:
                            # Đóng lượt của người nói trước khi có sự kiện đổi người nói hoặc khoảng nghỉ dài
                            session.current_open_turn_id = None
                            session.current_open_speaker = None
                            session.current_open_raw_speaker = None


        # Phân bổ từ ngữ theo mốc thời gian thực (Timestamp-Based Alignment)
        # Giúp 100% bảo toàn âm thanh đầu / chữ đầu khi có speaker mới bắt đầu phát biểu
        words_for_alignment = session.all_words_history if session.all_words_history else (self.asr.extract_interim_words_with_timestamps(session.asr_stream) if session.asr_stream else [])
        if words_for_alignment:
            temp_spk_words: Dict[str, List[str]] = {}
            segs_to_search = session.all_confirmed_segs if getattr(session, "all_confirmed_segs", None) else session.confirmed_segs
            for w in words_for_alignment:
                t = w["start"]
                best_spk = None
                for seg in segs_to_search:
                    if (seg.start - 0.25) <= t <= (seg.end + 0.25):
                        best_spk = seg.speaker
                        break
                if best_spk is None:
                    # Nếu từ mới xuất hiện trong 3s gần nhất (đuôi đang nói dở)
                    if t >= (session.duration_seconds - 3.0):
                        best_spk = session.active_speaker_id
                    elif segs_to_search:
                        # Từ cũ trong quá khứ: gán vào phân đoạn gần nhất theo mốc thời gian t, không gán bừa cho active_speaker hiện tại
                        closest_seg = min(segs_to_search, key=lambda s: min(abs(s.start - t), abs(s.end - t)))
                        best_spk = closest_seg.speaker
                    else:
                        best_spk = session.active_speaker_id

                if best_spk not in temp_spk_words:
                    temp_spk_words[best_spk] = []
                temp_spk_words[best_spk].append(w["word"])

            session.speaker_words = temp_spk_words
        elif partial_text:
            # Dự phòng nếu chưa có timestamps chi tiết
            current_words = [w.lower() for w in partial_text.split()]
            session.speaker_words[session.active_speaker_id] = current_words

        # Chuẩn bị danh sách nhiều người nói cho UI
        inactivity_timeout = float(opts.get("speaker_inactivity_timeout_s", self.config.speaker_inactivity_timeout_s))
        max_words = int(opts.get("max_recent_words_per_speaker", self.config.max_recent_words_per_speaker))

        all_spk_keys = set(session.speaker_embeddings.keys())
        all_spk_keys.update(session.speaker_identities.keys())
        all_spk_keys.update(session.speaker_last_active_time.keys())
        all_spk_keys.update(session.speaker_words.keys())
        for t in session.committed_turns:
            if t.get("raw_speaker"):
                all_spk_keys.add(t["raw_speaker"])
        if session.active_speaker_id:
            all_spk_keys.add(session.active_speaker_id)

        all_speakers_keys = sorted(
            list(all_spk_keys),
            key=lambda x: (int(x.replace("speaker_", "")) if x.replace("speaker_", "").isdigit() else 999, x)
        )

        speakers_list = []
        for spk in all_speakers_keys:
            name, score = self._get_speaker_display_identity(session, spk)
            last_audio_t = getattr(session, "speaker_last_active_audio_time", {}).get(spk, session.duration_seconds)
            last_wall_t = session.speaker_last_active_time.get(spk, now)
            audio_ago = max(0.0, session.duration_seconds - last_audio_t)
            wall_ago = max(0.0, now - last_wall_t)
            last_ago = min(audio_ago, wall_ago) if wall_ago > 0 else audio_ago
            is_speaking = (spk == session.active_speaker_id) and (last_ago <= 1.5)
            is_expired = last_ago > inactivity_timeout

            # Lấy tất cả các lượt phát biểu đã chốt của người này
            spk_committed_turns = [
                t for t in session.committed_turns
                if t.get("raw_speaker") == spk or t.get("speaker") == name
            ]
            latest_turn_text = spk_committed_turns[-1].get("text", "") if spk_committed_turns else ""
            committed_words_count = sum(len(t.get("raw_text", "").split()) for t in spk_committed_turns)

            if is_speaking:
                # Người này đang phát biểu trực tiếp: Chỉ lấy từ mới kể từ mốc chốt gần nhất
                last_commit_time = (session.last_committed_sample / sample_rate) if session.last_committed_sample > 0 else 0.0
                uncommitted_words = [
                    w["word"] for w in session.all_words_history
                    if w.get("start", 0.0) >= max(0.0, last_commit_time - 0.15)
                ]

                curr_turn_text = ""
                if spk_committed_turns:
                    last_turn = spk_committed_turns[-1]
                    if abs(last_turn.get("end", 0.0) - last_commit_time) <= 1.0:
                        curr_turn_text = last_turn.get("raw_text", "")

                if curr_turn_text and uncommitted_words:
                    combined_raw = merge_transcripts_without_overlap(curr_turn_text, " ".join(uncommitted_words))
                elif curr_turn_text:
                    combined_raw = curr_turn_text
                elif uncommitted_words:
                    combined_raw = " ".join(uncommitted_words)
                else:
                    combined_raw = ""

                if combined_raw:
                    words = combined_raw.split()
                    if max_words > 0 and len(words) > max_words:
                        words = words[-max_words:]
                    display_spk_text = format_natural_text(" ".join(words))
                elif latest_turn_text:
                    display_spk_text = latest_turn_text
                elif spk in session.speaker_words and session.speaker_words[spk]:
                    words = session.speaker_words[spk]
                    if max_words > 0 and len(words) > max_words:
                        words = words[-max_words:]
                    display_spk_text = format_natural_text(" ".join(words))
                else:
                    display_spk_text = ""

                total_spk_words = committed_words_count + len(uncommitted_words)
                if total_spk_words == 0 and spk in session.speaker_words:
                    total_spk_words = len(session.speaker_words[spk])
            else:
                # Người này đang tạm dừng nói -> hiển thị câu phát biểu hoàn chỉnh gần nhất của họ
                if latest_turn_text:
                    words = latest_turn_text.split()
                    if max_words > 0 and len(words) > max_words:
                        display_spk_text = format_natural_text(" ".join(words[-max_words:]))
                    else:
                        display_spk_text = latest_turn_text
                elif spk in session.speaker_words and session.speaker_words[spk]:
                    words = session.speaker_words[spk]
                    if max_words > 0 and len(words) > max_words:
                        words = words[-max_words:]
                    display_spk_text = format_natural_text(" ".join(words))
                else:
                    display_spk_text = ""

                total_spk_words = committed_words_count
                if total_spk_words == 0 and spk in session.speaker_words:
                    total_spk_words = len(session.speaker_words[spk])

            speakers_list.append({
                "speaker_id": spk,
                "name": name,
                "score": round(score, 3),
                "is_speaking": is_speaking,
                "is_expired": is_expired,
                "last_active_ago": round(last_ago, 1),
                "recent_text": display_spk_text,
                "total_words": total_spk_words
            })

        return {
            "type": "partial",
            "session_id": session_id,
            "text": format_natural_text(partial_text),
            "duration": round(session.duration_seconds, 2),
            "realtime_diarization": True,
            "active_speaker": session.active_identity,
            "active_speaker_id": session.active_speaker_id,
            "speakers": speakers_list,
            "turn_event": turn_event,
            "committed_turns": [
                {
                    "turn_id": t["turn_id"],
                    "speaker": t["speaker"],
                    "raw_speaker": t.get("raw_speaker", "speaker_0"),
                    "score": t.get("score", 1.0),
                    "text": t["text"],
                    "start": t["start"],
                    "end": t["end"],
                }
                for t in session.committed_turns
            ]
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

        enable_2pass = opts.get("enable_2pass_streaming", self.config.enable_2pass_streaming)
        # Nếu phiên ngắn không kịp commit lượt nào, chạy offline ASR giải mã trọn vẹn
        if enable_2pass and len(session.committed_turns) == 0 and self.offline_asr.enabled and len(clean_audio) >= int(0.3 * 16000):
            try:
                full_res = self.offline_asr.process_full_audio(clean_audio, sample_rate=16000)
                full_raw = clean_asr_hallucinations((full_res.get("text") or "").strip())
                if full_raw:
                    spk_id = session.active_speaker_id
                    display_name, score = self._get_speaker_display_identity(session, spk_id)
                    session.committed_turns.append({
                        "turn_id": f"turn_single_{uuid.uuid4().hex[:4]}",
                        "speaker": display_name,
                        "raw_speaker": spk_id,
                        "score": round(score, 3),
                        "raw_text": full_raw,
                        "text": format_natural_text(full_raw),
                        "start": 0.0,
                        "end": round(total_duration, 2),
                    })
                    session.last_committed_sample = len(clean_audio)
            except Exception as e:
                print(f"[Instant Finalize] Lỗi giải mã toàn bộ file ngắn: {e}")

        can_instant_finalize = enable_2pass and len(session.committed_turns) > 0

        if can_instant_finalize:
            # -----------------------------------------------------------------
            # CHẾ ĐỘ 1: KẾ THỪA TRỰC TIẾP TỪ REALTIME 2-PASS (INSTANT FINALIZE < 200ms)
            # Tái sử dụng 100% committed_turns và danh tính đã khóa trong lúc realtime!
            # -----------------------------------------------------------------
            if session.asr_stream is not None:
                try:
                    self.asr.finish_stream(session.asr_stream)
                except Exception:
                    pass

            # Xử lý phần âm thanh đuôi còn lại (Tail Audio) bằng ZipFormer-150M nếu có
            if self.offline_asr.enabled and len(clean_audio) > session.last_committed_sample:
                rem_pcm = clean_audio[session.last_committed_sample :]
                if len(rem_pcm) >= int(0.25 * 16000):
                    try:
                        rem_res = self.offline_asr.process_full_audio(rem_pcm, sample_rate=16000)
                        rem_text = (rem_res.get("text") or "").strip()
                        rem_text = clean_asr_hallucinations(rem_text)
                        if rem_text:
                            spk_id = session.active_speaker_id
                            display_name, score = self._get_speaker_display_identity(session, spk_id)
                            # Nếu cùng người nói với turn cuối thì gộp vào
                            if (
                                session.committed_turns
                                and session.current_open_raw_speaker == spk_id
                            ):
                                session.committed_turns[-1]["raw_text"] = merge_transcripts_without_overlap(session.committed_turns[-1]["raw_text"], rem_text)
                                session.committed_turns[-1]["end"] = round(total_duration, 2)
                            else:
                                session.committed_turns.append({
                                    "turn_id": f"turn_final_{uuid.uuid4().hex[:4]}",
                                    "speaker": display_name,
                                    "raw_speaker": spk_id,
                                    "score": round(score, 3),
                                    "raw_text": rem_text,
                                    "text": format_natural_text(rem_text),
                                    "start": round(session.last_committed_sample / 16000.0, 2),
                                    "end": round(total_duration, 2),
                                    })
                    except Exception as rem_err:
                        print(f"[Instant Finalize] Lỗi xử lý đoạn audio đuôi: {rem_err}")

            # Tái hợp nhất người nói (Speaker Consolidation): Gộp các speaker có giọng tương đồng (similarity >= 0.46)
            if len(session.speaker_embeddings) > 1:
                spk_keys = list(session.speaker_embeddings.keys())
                canonical_map = {k: k for k in spk_keys}
                for i in range(len(spk_keys)):
                    for j in range(i + 1, len(spk_keys)):
                        k1 = spk_keys[i]
                        k2 = spk_keys[j]
                        c1 = canonical_map[k1]
                        c2 = canonical_map[k2]
                        if c1 == c2:
                            continue
                        sim = float(np.dot(session.speaker_embeddings[c1], session.speaker_embeddings[c2]))
                        if sim >= 0.46:
                            for k in spk_keys:
                                if canonical_map[k] == c2:
                                    canonical_map[k] = c1

                # Cập nhật raw_speaker
                for turn in session.committed_turns:
                    raw_spk = turn.get("raw_speaker", "speaker_0")
                    if raw_spk in canonical_map:
                        turn["raw_speaker"] = canonical_map[raw_spk]

                # Gộp các turns liền kề của cùng một người nói sau khi hợp nhất
                merged_turns = []
                for turn in session.committed_turns:
                    if merged_turns and merged_turns[-1]["raw_speaker"] == turn["raw_speaker"]:
                        prev = merged_turns[-1]
                        prev["raw_text"] = merge_transcripts_without_overlap(prev["raw_text"], turn["raw_text"])
                        prev["end"] = max(prev["end"], turn["end"])
                    else:
                        merged_turns.append(turn)
                session.committed_turns = merged_turns

            # Kế thừa các lượt nói từ Realtime & Hậu xử lý ViBERT-capu
            formatted_segments = []
            final_turn_texts = []
            for turn in session.committed_turns:
                raw_spk = turn.get("raw_speaker", "speaker_0")
                display_name, score = self._get_speaker_display_identity(session, raw_spk)
                turn["speaker"] = display_name
                turn["score"] = round(score, 3)

                raw_t = clean_asr_hallucinations(turn.get("raw_text", "")).strip()
                raw_t = deduplicate_consecutive_phrases(raw_t)
                if not raw_t:
                    continue

                if opts.get("punctuation", True) and self.punctuation.enabled:
                    punc_text = self.punctuation.punctuate(raw_t)
                else:
                    punc_text = format_natural_text(raw_t)

                punc_text = deduplicate_consecutive_phrases(punc_text)

                turn["text"] = punc_text
                final_turn_texts.append(punc_text)
                formatted_segments.append({
                    "turn_id": turn.get("turn_id"),
                    "start": turn["start"],
                    "end": turn["end"],
                    "speaker": turn["speaker"],
                    "raw_speaker": raw_spk,
                    "score": turn.get("score", 1.0),
                    "text": punc_text
                })

            final_text = " ".join(final_turn_texts)
            final_text = deduplicate_consecutive_phrases(final_text)
            raw_text = " ".join([t.get("raw_text", "") for t in session.committed_turns])
            raw_text = deduplicate_consecutive_phrases(raw_text)

        else:
            # -----------------------------------------------------------------
            # CHẾ ĐỘ 2 (FALLBACK): Dành cho phiên không bật 2-Pass hoặc không có committed_turns
            # -----------------------------------------------------------------
            raw_text = ""
            words = []
            if opts.get("asr", True) and self.offline_asr.enabled and len(clean_audio) > 0:
                stream_info = self.offline_asr.process_full_audio(clean_audio, sample_rate=16000)
                raw_text = stream_info.get("text", "")
                tokens = stream_info.get("tokens", [])
                timestamps = stream_info.get("timestamps", [])
                if tokens and timestamps:
                    words = self.asr.extract_words_from_tokens(tokens, timestamps)
            elif opts.get("asr", True) and self.asr.enabled and session.asr_stream is not None:
                stream_info = self.asr.finish_stream_with_timestamps(session.asr_stream)
                raw_text = stream_info.get("text", "")
                tokens = stream_info.get("tokens", [])
                timestamps = stream_info.get("timestamps", [])
                if tokens and timestamps:
                    words = self.asr.extract_words_from_tokens(tokens, timestamps)
                if session.all_words_history and len(session.all_words_history) > len(words) + 10:
                    words = session.all_words_history.copy()
                    raw_text = " ".join([w["word"] for w in words])
                elif len(session.current_partial_text) > len(raw_text) + 20:
                    raw_text = session.current_partial_text
            else:
                raw_text = session.current_partial_text

            if not raw_text.strip() and session.current_partial_text:
                raw_text = session.current_partial_text

            if opts.get("punctuation", True) and self.punctuation.enabled and raw_text:
                final_text = self.punctuation.punctuate(raw_text)
            else:
                final_text = format_natural_text(raw_text)

            # Phân đoạn người nói (SortFormer / Nemotron Diarization)
            if opts.get("diarization", True) and self.diarization.enabled and len(clean_audio) > 0:
                d_col = opts.get("diarization_collar", self.config.diarization_collar)
                d_min = opts.get("diarization_min_duration", self.config.diarization_min_duration)
                segments = self.diarization.diarize_audio(clean_audio, sample_rate=16000, collar=d_col, min_duration=d_min)
            else:
                segments = [SpeakerSegment(start=0.0, end=round(total_duration, 3), speaker="speaker_0", confidence=1.0)]

            # Định danh người nói (TitaNet Speaker Verification)
            if opts.get("verification", True) and self.verification.enabled and len(clean_audio) > 0:
                v_th = opts.get("verification_threshold", self.config.verification_threshold)
                id_map = self.verification.resolve_identities(clean_audio, segments, sample_rate=16000, threshold=v_th)
            else:
                unique_spks = set(s.speaker for s in segments)
                id_map = {spk: (spk, 1.0) for spk in unique_spks}

            # Căn chỉnh văn bản với người nói
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


        # Tóm tắt nội dung tự động bằng Qwen2.5-1.5B (nếu được kích hoạt)
        summary_text = ""
        if opts.get("summary", True) and self.summary.enabled and (final_text or formatted_segments):
            try:
                summary_text = self.summary.summarize(text=final_text, segments=formatted_segments)
            except Exception as sum_err:
                print(f"[PipelineManager] Lỗi khi tóm tắt phiên online: {sum_err}")
                summary_text = f"Không thể tạo tóm tắt: {sum_err}"

        result = {
            "type": "final",
            "source": "mic",
            "session_id": session_id,
            "total_duration_s": round(total_duration, 2),
            "raw_text": format_natural_text(raw_text),
            "final_text": final_text,
            "segments": formatted_segments,
            "summary": summary_text,
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
                    "total_steps": 7,
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
        if sample_rate != 16000 and len(audio_pcm) > 0:
            try:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(sample_rate), 16000)
                audio_pcm = resample_poly(audio_pcm, 16000 // g, int(sample_rate) // g).astype(np.float32)
                sample_rate = 16000
            except Exception:
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
            "summary": self.config.enable_summary,
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
        if opts.get("asr", True) and self.offline_asr.enabled and total_samples > 0:
            report(3, "asr", 26, "Đang giải mã Offline ZipFormer (150M) siêu chính xác...")
            
            stream_info = self.offline_asr.process_full_audio(clean_audio, sample_rate=sample_rate)
            raw_text = stream_info.get("text", "")
            tokens = stream_info.get("tokens", [])
            timestamps = stream_info.get("timestamps", [])
            
            words = self.asr.extract_words_from_tokens(tokens, timestamps) if tokens else []
            word_count = len(raw_text.split()) if raw_text else 0
            report(3, "asr", 60, f"Nhận dạng ASR hoàn tất! ({word_count} từ)")
        else:
            words = []
            report(3, "asr", 60, "Bỏ qua nhận dạng ASR (Module đang tắt)", skipped=True)

        # BƯỚC 4: Phân đoạn người nói (60% -> 75%)
        check_abort()
        if opts.get("diarization", True) and self.diarization.enabled and len(clean_audio) > 0:
            diar_model_name = "Nemotron-3" if "nemotron" in self.diarization.model_name.lower() else "SortFormer"
            report(4, "diarization", 65, f"Đang phân đoạn người nói (NVIDIA {diar_model_name})...")
            d_col = opts.get("diarization_collar", self.config.diarization_collar)
            d_min = opts.get("diarization_min_duration", self.config.diarization_min_duration)
            segments = self.diarization.diarize_audio(clean_audio, sample_rate=16000, collar=d_col, min_duration=d_min)
            report(4, "diarization", 75, f"Phân đoạn người nói thành công! Tìm thấy {len(segments)} phân đoạn phát biểu.")
        else:
            segments = [SpeakerSegment(start=0.0, end=round(duration_s, 3), speaker="speaker_0", confidence=1.0)]
            report(4, "diarization", 75, "Bỏ qua phân đoạn người nói (Module đang tắt)", skipped=True)

        # BƯỚC 5: Định danh người nói TitaNet (75% -> 85%)
        check_abort()
        if opts.get("verification", True) and self.verification.enabled and len(clean_audio) > 0:
            report(5, "verification", 78, "Đang so khớp nhận diện danh tính người nói (NVIDIA TitaNet)...")
            v_th = opts.get("verification_threshold", self.config.verification_threshold)
            id_map = self.verification.resolve_identities(clean_audio, segments, sample_rate=16000, threshold=v_th)
            report(5, "verification", 85, f"Định danh giọng nói hoàn tất! Đã tra cứu {len(id_map)} người nói.")
        else:
            unique_spks = set(s.speaker for s in segments)
            id_map = {spk: (spk, 1.0) for spk in unique_spks}
            report(5, "verification", 85, "Bỏ qua định danh người nói (Module đang tắt)", skipped=True)

        # BƯỚC 6: Chuẩn hóa dấu câu ViBERT-capu (85% -> 92%)
        check_abort()
        if opts.get("punctuation", True) and self.punctuation.enabled and raw_text:
            report(6, "punctuation", 88, "Đang thêm dấu câu & viết hoa chuẩn hóa (ViBERT-capu ONNX)...")
            final_text = self.punctuation.punctuate(raw_text)
            report(6, "punctuation", 92, "Chuẩn hóa văn bản hoàn tất!")
        else:
            final_text = format_natural_text(raw_text)
            report(6, "punctuation", 92, "Bỏ qua chuẩn hóa dấu câu", skipped=True)

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

        # BƯỚC 7: Tóm tắt nội dung tự động (Qwen2.5-1.5B) (92% -> 100%)
        check_abort()
        summary_text = ""
        if opts.get("summary", True) and self.summary.enabled and (final_text or formatted_segments):
            report(7, "summary", 94, "Đang tóm tắt nội dung bằng mô hình Qwen2.5-1.5B-Instruct...")
            try:
                summary_text = self.summary.summarize(text=final_text, segments=formatted_segments)
                report(7, "summary", 100, "Tóm tắt nội dung thành công!")
            except Exception as sum_err:
                print(f"[PipelineManager] Lỗi khi tóm tắt offline: {sum_err}")
                summary_text = f"Không thể tạo tóm tắt: {sum_err}"
                report(7, "summary", 100, "Tóm tắt nội dung thất bại", skipped=True)
        else:
            report(7, "summary", 100, "Bỏ qua tóm tắt nội dung (Module đang tắt)", skipped=True)

        check_abort()
        report(7, "completed", 100, "Xử lý toàn bộ video/audio thành công!")

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
            "summary": summary_text,
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
                t_mid = (w["start"] + w["end"]) / 2.0
                best_seg = None

                # Ưu tiên 1: Tìm phân đoạn chứa trọn vẹn điểm giữa của từ
                for seg in sorted_segs:
                    if seg.start <= t_mid <= seg.end:
                        best_seg = seg
                        break
                
                # Ưu tiên 2: Nếu từ nằm ở khoảng trống giữa 2 phân đoạn, chọn phân đoạn gần nhất
                if best_seg is None:
                    min_dist = float("inf")
                    for seg in sorted_segs:
                        # Tính khoảng cách từ t_mid đến khoảng [seg.start, seg.end]
                        if t_mid < seg.start:
                            dist = seg.start - t_mid
                        elif t_mid > seg.end:
                            dist = t_mid - seg.end
                        else:
                            dist = 0.0
                            
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
                if real_name in ("Unknown", "Chưa đăng ký"):
                    spk_num = raw_spk.replace("speaker_", "")
                    real_name = f"Người nói {int(spk_num) + 1 if spk_num.isdigit() else raw_spk} (Chưa đăng ký)"

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
