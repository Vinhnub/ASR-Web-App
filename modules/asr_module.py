# -*- coding: utf-8 -*-
"""
Module Nhận dạng Giọng nói Thời gian thực (ZipFormer Streaming ASR + Cache).
Sử dụng mô hình Zipformer Transducer thông qua sherpa-onnx.
Hỗ trợ Contextual Biasing (In-Decoding Hotwords) trực tiếp trong quá trình giải mã Modified Beam Search,
quản lý phiên đa kết nối (multi-session streams) và cơ chế Bật / Tắt (Bypass).
"""

import os
import sys
import glob
import json
import numpy as np
from typing import Optional, Dict, Tuple, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class ASRModule:
    def __init__(
        self,
        model_dir: str,
        bpe_vocab_path: Optional[str] = None,
        hotwords_file: Optional[str] = None,
        hotwords_score: float = 2.0,
        decoding_method: str = "modified_beam_search",
        max_active_paths: int = 4,
        num_threads: int = 4,
        provider: str = "cpu",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model_dir = model_dir
        self.bpe_vocab_path = bpe_vocab_path
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self.num_threads = num_threads
        self.provider = provider
        self.recognizer = None

        if self.enabled:
            self._init_model()

    @staticmethod
    def find_model_files(model_dir: str):
        """Tìm các file ONNX và tokens cần thiết cho Zipformer Online Transducer."""
        def find_first(patterns):
            for pat in patterns:
                matches = glob.glob(os.path.join(model_dir, pat))
                if not matches:
                    matches = glob.glob(os.path.join(model_dir, "**", pat), recursive=True)
                if matches:
                    return matches[0]
            return None

        encoder = find_first(["*encoder*chunk*left*.onnx", "*encoder*.onnx"])
        decoder = find_first(["*decoder*chunk*left*.onnx", "*decoder*.onnx"])
        joiner = find_first(["*joiner*chunk*left*.onnx", "*joiner*.onnx"])
        tokens = find_first(["*tokens*.txt", "*config*.json", "*bpe*.model"])
        return encoder, decoder, joiner, tokens

    def _init_model(self):
        try:
            import sherpa_onnx
        except ImportError:
            print("[ASRModule] LỖI: Chưa cài đặt 'sherpa-onnx'. Vui lòng chạy: pip install sherpa-onnx")
            return

        print(f"[ASRModule] Đang khởi tạo ZipFormer Streaming ASR từ '{self.model_dir}'...")
        encoder, decoder, joiner, tokens = self.find_model_files(self.model_dir)
        if not (encoder and decoder and joiner and tokens):
            print(f"[ASRModule] CẢNH BÁO: Không tìm đủ các file thành phần ONNX trong '{self.model_dir}'")
            return

        kwargs = {
            "tokens": tokens,
            "encoder": encoder,
            "decoder": decoder,
            "joiner": joiner,
            "num_threads": self.num_threads,
            "sample_rate": 16000,
            "feature_dim": 80,
            "decoding_method": self.decoding_method,
            "provider": self.provider
        }

        # Thiết lập In-Decoding Contextual Biasing với BPE Vocab và Hotwords
        if self.decoding_method == "modified_beam_search":
            kwargs["max_active_paths"] = self.max_active_paths
            kwargs["hotwords_score"] = self.hotwords_score
            kwargs["modeling_unit"] = "bpe"
            if self.bpe_vocab_path and os.path.exists(self.bpe_vocab_path):
                kwargs["bpe_vocab"] = self.bpe_vocab_path
                print(f"[ASRModule] Đã nạp BPE Vocab cho Hotwords Biasing: '{os.path.basename(self.bpe_vocab_path)}'")
            if self.hotwords_file and os.path.exists(self.hotwords_file):
                kwargs["hotwords_file"] = self.hotwords_file
                print(f"[ASRModule] Đã nạp Hotwords File: '{os.path.basename(self.hotwords_file)}' (Bonus Score: +{self.hotwords_score})")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        print(f"[ASRModule] Khởi tạo ZipFormer Streaming ASR thành công! (Decoding: {self.decoding_method})")

    def reload_hotwords(self, hotwords_file: Optional[str] = None):
        """Tải lại mô hình khi danh sách hotwords được cập nhật."""
        if hotwords_file:
            self.hotwords_file = hotwords_file
        if self.enabled:
            self._init_model()

    def create_stream(self, hotwords: Optional[str] = None):
        """
        Tạo một luồng nhận diện mới cho mỗi client / phiên kết nối.
        - Nếu hotwords là chuỗi (kể cả rỗng ""), truyền trực tiếp vào stream để ghi đè (override).
          Ví dụ: hotwords="" sẽ tắt hotwords biasing cho stream này.
        - Nếu hotwords is None, stream sử dụng danh sách mặc định từ recognizer.
        """
        if not self.enabled or self.recognizer is None:
            return None
        if hotwords is not None:
            return self.recognizer.create_stream(hotwords=hotwords)
        return self.recognizer.create_stream()

    def process_chunk(self, stream, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Nạp một chunk âm thanh vào stream, giải mã và trả về văn bản tích lũy hiện tại (Interim Text).
        """
        if not self.enabled or self.recognizer is None or stream is None:
            return ""

        if audio_chunk is None or len(audio_chunk) == 0:
            res = self.recognizer.get_result(stream)
            return res.text if hasattr(res, "text") else str(res)

        # Chuẩn hóa kiểu float32
        chunk_f32 = audio_chunk.astype(np.float32)
        stream.accept_waveform(sample_rate, chunk_f32)

        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

        res = self.recognizer.get_result(stream)
        return res.text if hasattr(res, "text") else str(res)

    def extract_interim_words_with_timestamps(self, stream) -> List[Dict[str, Any]]:
        """
        Trích xuất danh sách các từ kèm mốc thời gian tức thì (start, end) từ luồng stream đang chạy.
        """
        if not self.enabled or self.recognizer is None or stream is None:
            return []
        try:
            raw_json = self.recognizer.get_result_as_json_string(stream)
            data = json.loads(raw_json)
            tokens = data.get("tokens", [])
            timestamps = data.get("timestamps", [])
            if tokens and timestamps:
                return self.extract_words_from_tokens(tokens, timestamps)
        except Exception:
            pass
        return []

    def finish_stream_with_timestamps(self, stream) -> Dict[str, Any]:
        """
        Báo kết thúc luồng nhận dạng và thu kết quả cuối cùng kèm tokens và timestamps.
        """
        if not self.enabled or self.recognizer is None or stream is None:
            return {"text": "", "tokens": [], "timestamps": []}

        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

        try:
            raw_json = self.recognizer.get_result_as_json_string(stream)
            data = json.loads(raw_json)
            text = data.get("text", "")
            tokens = data.get("tokens", [])
            timestamps = data.get("timestamps", [])
            return {
                "text": text,
                "tokens": tokens,
                "timestamps": timestamps
            }
        except Exception:
            res = self.recognizer.get_result(stream)
            text = res.text if hasattr(res, "text") else str(res)
            return {"text": text, "tokens": [], "timestamps": []}

    def finish_stream(self, stream) -> str:
        """
        Báo kết thúc luồng nhận dạng và thu kết quả cuối cùng (văn bản).
        """
        return self.finish_stream_with_timestamps(stream)["text"]

    @staticmethod
    def extract_words_from_tokens(tokens: List[str], timestamps: List[float]) -> List[Dict[str, Any]]:
        """
        Gom nhóm các BPE tokens thành từng từ hoàn chỉnh kèm thời điểm bắt đầu và kết thúc (giây).
        Tokens bắt đầu bằng ' ' hoặc ' ' (BPE whitespace marker) là điểm bắt đầu của từ mới.
        """
        words = []
        curr_word = ""
        curr_start = 0.0

        for tok, t in zip(tokens, timestamps):
            is_new_word = tok.startswith(" ") or tok.startswith(" ") or tok.startswith("_")
            clean_tok = tok.lstrip(" ").lstrip(" ").lstrip("_")

            if is_new_word and curr_word:
                words.append({
                    "word": curr_word.strip().lower(),
                    "start": round(curr_start, 3),
                    "end": round(t, 3)
                })
                curr_word = clean_tok
                curr_start = t
            else:
                if not curr_word:
                    curr_start = t
                curr_word += clean_tok

        if curr_word.strip():
            end_t = timestamps[-1] if timestamps else curr_start + 0.3
            words.append({
                "word": curr_word.strip().lower(),
                "start": round(curr_start, 3),
                "end": round(end_t, 3)
            })

        return words

    def transcribe_full_audio_with_timestamps(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        chunk_size_s: float = 0.32,
        hotwords: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mô phỏng streaming trên toàn bộ mảng âm thanh hoàn chỉnh để trả về văn bản kèm tokens và timestamps.
        """
        if not self.enabled or self.recognizer is None or len(audio_pcm) == 0:
            return {"text": "", "tokens": [], "timestamps": []}

        stream = self.create_stream(hotwords=hotwords)
        chunk_samples = int(chunk_size_s * sample_rate)
        total_len = len(audio_pcm)

        for i in range(0, total_len, chunk_samples):
            chunk = audio_pcm[i : i + chunk_samples]
            self.process_chunk(stream, chunk, sample_rate)

        return self.finish_stream_with_timestamps(stream)

    def transcribe_full_audio(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        chunk_size_s: float = 0.32,
        hotwords: Optional[str] = None
    ) -> str:
        """
        Mô phỏng streaming trên toàn bộ mảng âm thanh hoàn chỉnh để trả về kết quả ASR (văn bản).
        Hỗ trợ truyền hotwords động cho từng phân đoạn nói.
        """
        return self.transcribe_full_audio_with_timestamps(
            audio_pcm=audio_pcm,
            sample_rate=sample_rate,
            chunk_size_s=chunk_size_s,
            hotwords=hotwords
        )["text"]

    def set_parameters(
        self,
        decoding_method: Optional[str] = None,
        max_active_paths: Optional[int] = None,
        hotwords_score: Optional[float] = None,
    ):
        """Cập nhật linh hoạt các thông số giải mã ZipFormer."""
        reinit = False
        if decoding_method is not None and decoding_method != self.decoding_method:
            self.decoding_method = str(decoding_method)
            reinit = True
        if max_active_paths is not None and int(max_active_paths) != self.max_active_paths:
            self.max_active_paths = int(max_active_paths)
            reinit = True
        if hotwords_score is not None and float(hotwords_score) != self.hotwords_score:
            self.hotwords_score = float(hotwords_score)
            reinit = True

        if reinit and self.enabled:
            print(f"[ASRModule] Áp dụng lại mô hình với tham số: method={self.decoding_method}, paths={self.max_active_paths}, hw_score={self.hotwords_score}")
            self._init_model()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if self.enabled and self.recognizer is None:
            self._init_model()
