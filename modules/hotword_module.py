# -*- coding: utf-8 -*-
"""
Module Quản lý Từ khóa Chuyên ngành (Hotwords Manager) cho ZipFormer Contextual Biasing (Cách 1).
Quản lý danh sách hotwords trong file text để nhúng trực tiếp vào thuật toán Modified Beam Search của ZipFormer.
"""

import os
from typing import Dict, List, Optional, Any


class HotwordModule:
    def __init__(
        self,
        hotwords_file: str,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.hotwords_file = hotwords_file
        # Lưu trữ dict: phrase -> score (mặc định 2.0)
        self.hotwords: Dict[str, float] = {}

        self.load_hotwords()

    def load_hotwords(self):
        """Đọc danh sách hotwords từ file text (mỗi dòng 1 từ/cụm từ, có thể kèm :score)."""
        if not os.path.exists(self.hotwords_file):
            print(f"[HotwordModule] File '{self.hotwords_file}' chưa tồn tại -> Khởi tạo file mới.")
            self.hotwords = {
                "PARACETAMOL": 2.5,
                "SORTFORMER": 2.0,
                "TITANET": 2.0,
                "VIBERT": 2.0,
                "DEEPFILTER": 2.0,
                "DEEPFILTERNET": 2.0,
                "API": 2.0,
                "PYTHON": 1.5,
                "ONNX": 2.0,
                "ZIPFORMER": 2.0,
            }
            self.save_hotwords()
            return

        try:
            self.hotwords = {}
            with open(self.hotwords_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Format: PHRASE hoặc PHRASE :SCORE
                    if ":" in line:
                        parts = line.split(":", 1)
                        phrase = parts[0].strip().upper()
                        try:
                            score = float(parts[1].strip())
                        except ValueError:
                            score = 2.0
                    else:
                        phrase = line.strip().upper()
                        score = 2.0

                    if phrase:
                        self.hotwords[phrase] = score

            print(f"[HotwordModule] Đã nạp {len(self.hotwords)} từ khóa in-decoding cho ZipFormer từ '{os.path.basename(self.hotwords_file)}'.")
        except Exception as e:
            print(f"[HotwordModule] Cảnh báo lỗi đọc file hotwords: {e}")
            self.hotwords = {}

    def save_hotwords(self):
        """Lưu lại danh sách từ khóa ra file text."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.hotwords_file)), exist_ok=True)
            with open(self.hotwords_file, "w", encoding="utf-8") as f:
                for phrase, score in sorted(self.hotwords.items()):
                    if abs(score - 2.0) < 1e-4:
                        f.write(f"{phrase}\n")
                    else:
                        f.write(f"{phrase} :{score:.1f}\n")
        except Exception as e:
            print(f"[HotwordModule] Lỗi lưu file hotwords: {e}")

    def add_hotword(self, phrase: str, score: float = 2.0):
        """Thêm hoặc cập nhật một từ khóa chuyên ngành."""
        p = phrase.strip().upper()
        if p:
            self.hotwords[p] = float(score)
            self.save_hotwords()

    def remove_hotword(self, phrase: str) -> bool:
        """Xóa một từ khóa khỏi danh sách."""
        p = phrase.strip().upper()
        if p in self.hotwords:
            del self.hotwords[p]
            self.save_hotwords()
            return True
        return False

    def get_hotwords_list(self) -> List[Dict[str, Any]]:
        """Trả về danh sách phục vụ Web UI / REST API."""
        return [
            {"phrase": phrase, "score": score}
            for phrase, score in sorted(self.hotwords.items())
        ]

    def get_hotwords_stream_str(self) -> str:
        """
        Trả về chuỗi hotwords định dạng cho `rec.create_stream(hotwords=...)`.
        Các cụm từ phân cách nhau bởi dấu '/' (chuẩn sherpa-onnx).
        Nếu module bị tắt hoặc danh sách trống, trả về chuỗi rỗng "".
        """
        if not self.enabled or not self.hotwords:
            return ""
        # Trong sherpa-onnx, danh sách phân cách bởi '/'
        return "/".join(self.hotwords.keys())

    def correct(self, text: str) -> str:
        """
        Pass-through function để tương thích ngược.
        Cách 1 (In-Decoding Hotwords) đã áp dụng trực tiếp trong lúc ZipFormer giải mã âm học,
        không còn phụ thuộc vào regex post-correction nữa.
        """
        return text

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
