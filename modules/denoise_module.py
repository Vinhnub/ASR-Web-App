# -*- coding: utf-8 -*-
"""
Module Khử nhiễu âm thanh thời gian thực (DeepFilterNet3).
Hỗ trợ chuyển đổi tần số mẫu và cơ chế Bật / Tắt linh hoạt (Bypass).
"""

import os
import sys
import numpy as np
from typing import Optional

# Thêm đường dẫn gốc để import NoiseFilter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from NoiseFilter.deepfilter_module import DeepFilter


class DenoiseModule:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "DeepFilterNet3",
        atten_lim_db: Optional[float] = 15.0,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.device = device
        self.model_name = model_name
        self.atten_lim_db = atten_lim_db
        self.filter: Optional[DeepFilter] = None

        if self.enabled:
            self._init_model()

    def _init_model(self):
        try:
            print(f"[DenoiseModule] Đang tải {self.model_name} trên thiết bị: {self.device.upper()}...")
            self.filter = DeepFilter(
                model_name=self.model_name,
                device=self.device,
                atten_lim_db=self.atten_lim_db,
                log_level="WARNING"
            )
            print(f"[DenoiseModule] Khởi tạo {self.model_name} thành công!")
        except Exception as e:
            print(f"[DenoiseModule] Cảnh báo không thể tải model DeepFilter: {e}")
            self.filter = None

    def process(self, audio_pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Khử nhiễu cho mảng âm thanh (1D float32 trong khoảng [-1.0, 1.0]).
        Chuẩn hóa theo đúng logic từ nhỏ đến lớn:
        - 0 dB: Không khử nhiễu (giữ nguyên vẹn 100% âm thanh gốc).
        - Càng lớn (10 -> 40 dB): Khử nhiễu càng mạnh.
        - >= 50 dB: Khử triệt để tối đa 100%.
        """
        if not self.enabled or self.filter is None or audio_pcm is None or len(audio_pcm) == 0:
            return audio_pcm

        # 0 dB: Không khử nhiễu, giữ nguyên vẹn âm gốc
        if self.atten_lim_db is not None and self.atten_lim_db <= 0:
            return audio_pcm

        # >= 50 dB: Khử sạch tối đa không giới hạn (Full suppression)
        lim = None if (self.atten_lim_db is not None and self.atten_lim_db >= 50) else self.atten_lim_db

        try:
            cleaned = self.filter.filter_audio(
                audio_pcm,
                orig_sr=sample_rate,
                target_sr=sample_rate,
                atten_lim_db=lim
            )
            return cleaned.astype(np.float32)
        except Exception as err:
            print(f"[DenoiseModule] Lỗi xử lý khử nhiễu: {err} -> Giữ nguyên âm thanh gốc.")
            return audio_pcm

    def set_parameters(
        self,
        atten_lim_db: Optional[float] = None,
    ):
        """Cập nhật linh hoạt thông số mức độ khử nhiễu (dB): 0 = không khử, càng lớn khử càng mạnh."""
        if atten_lim_db is not None:
            self.atten_lim_db = float(atten_lim_db)
            print(f"[DenoiseModule] Đã cập nhật tham số: atten_lim_db={self.atten_lim_db}")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if self.enabled and self.filter is None:
            self._init_model()
