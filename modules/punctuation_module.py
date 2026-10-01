# -*- coding: utf-8 -*-
"""
Module Phục hồi Dấu câu & Viết hoa Tự động (Automatic Punctuation - ViBERT-capu).
Sử dụng mô hình ONNX Seq2Labels của ViBERT-capu để thêm dấu chấm, phẩy, hỏi và viết hoa.
Hỗ trợ cơ chế Bật / Tắt (Bypass).
"""

import os
import sys
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Automatic_Punctuation.runners import ViBERTOnnxRunner


class PunctuationModule:
    def __init__(
        self,
        variant: str = "int8",
        device: str = "cpu",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.variant = variant
        self.device = device
        self.runner: Optional[ViBERTOnnxRunner] = None

        if self.enabled:
            self._init_model()

    def _init_model(self):
        try:
            print(f"[PunctuationModule] Đang tải ViBERT-capu ONNX ({self.variant.upper()}) trên {self.device.upper()}...")
            self.runner = ViBERTOnnxRunner(
                variant=self.variant,
                device=self.device
            )
            print(f"[PunctuationModule] Khởi tạo ViBERT-capu thành công!")
        except Exception as e:
            print(f"[PunctuationModule] Cảnh báo không thể tải ViBERT-capu: {e}")
            self.runner = None

    def punctuate(self, text: str) -> str:
        """
        Thêm dấu câu và viết hoa cho văn bản tiếng Việt.
        Nếu cờ enabled=False, trả về văn bản gốc không thay đổi.
        """
        if not self.enabled or self.runner is None or not text or not text.strip():
            return text

        try:
            return self.runner.predict(text.strip().lower())
        except Exception as e:
            print(f"[PunctuationModule] Lỗi khi thêm dấu câu: {e} -> Giữ nguyên text gốc.")
            return text

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if self.enabled and self.runner is None:
            self._init_model()
