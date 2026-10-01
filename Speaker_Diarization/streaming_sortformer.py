# -*- coding: utf-8 -*-
"""
================================================================================
NVIDIA Streaming Sortformer - Real-time Speaker Diarization
================================================================================
Module cung cấp khả năng phân tách người nói theo thời gian thực (Online Diarization)
dựa trên kiến trúc NVIDIA Streaming Sortformer (FastConformer + Transformer Encoder
kết hợp Sort Loss và Arrival-Order Speaker Cache - AOSC).

Mô hình mặc định: nvidia/diar_streaming_sortformer_4spk-v2.1
Hỗ trợ:
- Diarization file âm thanh với cơ chế streaming nội tại của NeMo.
- Cấu hình linh hoạt các tham số streaming cache (chunk_len, fifo_len, spkcache_len).
- Lọc nhiễu, kết hợp các đoạn nói gần nhau (collar smoothing) và phát hiện Overlap.
- Xuất kết quả chuẩn hóa RTTM, JSON và hiển thị trực quan trên Console.
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

# Thiết lập UTF-8 cho Windows console để tránh lỗi Unicode
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

# Vá lỗi CPython TemporaryDirectory rmtree trên Windows khi file manifest.json bị khóa
if sys.platform == "win32":
    import tempfile
    import shutil
    @classmethod
    def _safe_rmtree(cls, name, ignore_errors=False):
        def _onerror(func, path, exc_info):
            if ignore_errors:
                return
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    pass
            else:
                for _ in range(5):
                    time.sleep(0.05)
                    try:
                        os.unlink(path)
                        return
                    except Exception:
                        pass
        try:
            shutil.rmtree(name, onerror=_onerror)
        except Exception:
            pass

    tempfile.TemporaryDirectory._rmtree = _safe_rmtree



@dataclass
class SpeakerSegment:
    """
    Đại diện cho một phân đoạn phát biểu của một người nói.
    """
    start: float           # Thời điểm bắt đầu (giây)
    end: float             # Thời điểm kết thúc (giây)
    speaker: str           # Nhãn người nói (ví dụ: spk_0, spk_1)
    confidence: float = 1.0  # Độ tin cậy (nếu có)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "speaker": self.speaker,
            "confidence": round(self.confidence, 4)
        }

    def to_rttm_line(self, session_id: str = "audio") -> str:
        """
        Định dạng theo chuẩn quốc tế RTTM (Rich Transcription Time Marked):
        SPEAKER <file-id> <channel> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
        """
        return f"SPEAKER {session_id} 1 {self.start:.3f} {self.duration:.3f} <NA> <NA> {self.speaker} <NA> <NA>"


class StreamingSortformerDiarizer:
    """
    Trình phân tách người nói thời gian thực NVIDIA Streaming Sortformer.
    """

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
        device: Optional[str] = None,
        chunk_len: int = 340,
        chunk_right_context: int = 40,
        fifo_len: int = 40,
        spkcache_update_period: int = 300,
        spkcache_len: int = 188,
        collar: float = 0.25,
        min_duration: float = 0.20,
    ):
        """
        Khởi tạo và cấu hình mô hình Streaming Sortformer.

        Tham số:
            model_name: Tên checkpoint HuggingFace hoặc đường dẫn file .nemo cục bộ
            device: 'cuda' hoặc 'cpu' (mặc định tự chọn CUDA nếu có)
            chunk_len: Kích thước khối frame xử lý (độ trễ chunk)
            chunk_right_context: Số frame ngữ cảnh tương lai đi kèm
            fifo_len: Số frame hàng đợi FIFO ngữ cảnh quá khứ
            spkcache_update_period: Chu kỳ cập nhật bộ đệm đặc trưng người nói (AOSC)
            spkcache_len: Dung lượng bộ đệm người nói
            collar: Khoảng cách tối đa (giây) giữa 2 đoạn cùng người nói để nối liền
            min_duration: Thời lượng tối thiểu (giây) của một đoạn nói để giữ lại
        """
        self.model_name = model_name
        self.chunk_len = chunk_len
        self.chunk_right_context = chunk_right_context
        self.fifo_len = fifo_len
        self.spkcache_update_period = spkcache_update_period
        self.spkcache_len = spkcache_len
        self.collar = collar
        self.min_duration = min_duration

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[StreamingSortformer] Thiết bị thực thi: {self.device.upper()}")
        self.model = self._load_model()

    def _load_model(self):
        """
        Tải mô hình Sortformer từ NeMo Toolkit.
        """
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            raise ImportError(
                "\n[LỖI] Chưa cài đặt thư viện NVIDIA NeMo Toolkit!\n"
                "Vui lòng cài đặt bằng lệnh:\n"
                "    pip install nemo_toolkit['asr']\n"
                "Hoặc tham khảo hướng dẫn tại: https://github.com/NVIDIA/NeMo\n"
            ) from e

        print(f"[StreamingSortformer] Đang tải mô hình '{self.model_name}'...")
        start_t = time.time()

        try:
            if os.path.isfile(self.model_name) and self.model_name.endswith(".nemo"):
                model = SortformerEncLabelModel.restore_from(restore_path=self.model_name)
            else:
                # Tải từ Hugging Face
                model = SortformerEncLabelModel.from_pretrained(model_name=self.model_name)
        except Exception as err:
            # Fallback sang checkpoint v2 nếu checkpoint v2.1 chưa khả dụng
            fallback_name = "nvidia/diar_streaming_sortformer_4spk-v2"
            if self.model_name != fallback_name:
                print(f"[Cảnh báo] Không thể tải '{self.model_name}'. Thử chuyển sang '{fallback_name}'...")
                model = SortformerEncLabelModel.from_pretrained(model_name=fallback_name)
                self.model_name = fallback_name
            else:
                raise err

        model = model.to(self.device)
        model.eval()

        # Cấu hình các tham số streaming trên sortformer_modules (nếu có)
        if hasattr(model, "sortformer_modules"):
            try:
                model.sortformer_modules.chunk_len = self.chunk_len
                model.sortformer_modules.chunk_right_context = self.chunk_right_context
                model.sortformer_modules.fifo_len = self.fifo_len
                model.sortformer_modules.spkcache_update_period = self.spkcache_update_period
                model.sortformer_modules.spkcache_len = self.spkcache_len
                if hasattr(model.sortformer_modules, "_check_streaming_parameters"):
                    model.sortformer_modules._check_streaming_parameters()
                print(
                    f"[StreamingSortformer] Đã áp dụng streaming cache: "
                    f"chunk_len={self.chunk_len}, fifo={self.fifo_len}, cache_len={self.spkcache_len}"
                )
            except Exception as conf_err:
                print(f"[StreamingSortformer] Cảnh báo cấu hình streaming modules: {conf_err}")

        elapsed = time.time() - start_t
        print(f"[StreamingSortformer] Tải mô hình thành công trong {elapsed:.2f}s!")
        return model

    def _parse_raw_segment(self, raw_seg: Union[str, tuple, list, object]) -> Optional[SpeakerSegment]:
        """
        Chuẩn hóa kết quả trả về từ NeMo diarize thành đối tượng SpeakerSegment.
        """
        try:
            # Trường hợp đối tượng có các thuộc tính start, end, speaker
            if hasattr(raw_seg, "start") and hasattr(raw_seg, "end"):
                speaker = getattr(raw_seg, "speaker", getattr(raw_seg, "label", "speaker_0"))
                conf = getattr(raw_seg, "confidence", 1.0)
                return SpeakerSegment(start=float(raw_seg.start), end=float(raw_seg.end), speaker=str(speaker), confidence=float(conf))

            # Trường hợp tuple hoặc list: [start, end, speaker, (confidence)]
            if isinstance(raw_seg, (list, tuple)):
                if len(raw_seg) >= 3:
                    conf = float(raw_seg[3]) if len(raw_seg) >= 4 else 1.0
                    return SpeakerSegment(start=float(raw_seg[0]), end=float(raw_seg[1]), speaker=str(raw_seg[2]), confidence=conf)

            # Trường hợp chuỗi văn bản phân tách bằng khoảng trắng: "0.00 1.50 speaker_0"
            if isinstance(raw_seg, str):
                parts = raw_seg.strip().split()
                if len(parts) >= 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    speaker = parts[2]
                    conf = float(parts[3]) if len(parts) >= 4 else 1.0
                    return SpeakerSegment(start=start, end=end, speaker=speaker, confidence=conf)

        except Exception:
            pass
        return None

    def postprocess_segments(
        self,
        segments: List[SpeakerSegment],
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> List[SpeakerSegment]:
        """
        Hậu xử lý danh sách phân đoạn:
        1. Sắp xếp theo thời gian bắt đầu.
        2. Gom cụm các đoạn liên tiếp cùng người nói nếu khoảng cách giữa chúng <= collar.
        3. Lọc bỏ các đoạn quá ngắn (< min_duration).
        """
        if not segments:
            return []

        collar = self.collar if collar is None else collar
        min_duration = self.min_duration if min_duration is None else min_duration

        # Sắp xếp theo thời gian bắt đầu
        sorted_segs = sorted(segments, key=lambda s: (s.speaker, s.start))

        # Gom các đoạn liền kề cùng một speaker
        merged_by_speaker: Dict[str, List[SpeakerSegment]] = {}
        for seg in sorted_segs:
            spk = seg.speaker
            if spk not in merged_by_speaker:
                merged_by_speaker[spk] = [seg]
            else:
                last_seg = merged_by_speaker[spk][-1]
                # Nếu đoạn tiếp theo bắt đầu gần kề hoặc gối đầu lên đoạn trước
                if seg.start <= last_seg.end + collar:
                    # Mở rộng đoạn trước
                    merged_by_speaker[spk][-1] = SpeakerSegment(
                        start=last_seg.start,
                        end=max(last_seg.end, seg.end),
                        speaker=spk,
                        confidence=(last_seg.confidence + seg.confidence) / 2.0
                    )
                else:
                    merged_by_speaker[spk].append(seg)

        # Gộp tất cả lại và sắp xếp theo thời gian thực
        all_merged: List[SpeakerSegment] = []
        for spk, seg_list in merged_by_speaker.items():
            for s in seg_list:
                if s.duration >= min_duration:
                    all_merged.append(s)

        all_merged.sort(key=lambda s: s.start)
        return all_merged

    def diarize_file(
        self,
        audio_path: str,
        batch_size: int = 1,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
        verbose: bool = True,
    ) -> List[SpeakerSegment]:
        """
        Thực hiện diarization trên file âm thanh hoàn chỉnh.

        Tham số:
            audio_path: Đường dẫn tới file âm thanh (.wav 16kHz mono khuyến nghị)
            batch_size: Kích thước batch
            collar: Ngưỡng nối đoạn gần nhau
            min_duration: Thời lượng tối thiểu giữ lại
            verbose: Bật/tắt hiển thị log ra màn hình

        Trả về:
            Danh sách SpeakerSegment đã được sắp xếp và hậu xử lý.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: '{audio_path}'")

        # Kiểm tra file rỗng hoặc hỏng (chỉ có WAV header 44 bytes)
        if os.path.getsize(audio_path) <= 100:
            if verbose:
                print(f"[StreamingSortformer] Cảnh báo: File rỗng hoặc không có dữ liệu âm thanh: '{audio_path}' -> Bỏ qua.")
            return []

        if verbose:
            print(f"[StreamingSortformer] Đang phân tách người nói: '{audio_path}'...")
        t0 = time.time()

        with torch.no_grad():
            try:
                results = self.model.diarize(
                    audio=[audio_path],
                    batch_size=batch_size,
                    num_workers=0,
                    verbose=False
                )
            except Exception as e:
                # Nếu gặp lỗi khóa file tạm thời trên Windows, thử lại sau 100ms
                if "WinError" in str(e) or "manifest" in str(e) or "directory name is invalid" in str(e).lower():
                    time.sleep(0.1)
                    results = self.model.diarize(
                        audio=[audio_path],
                        batch_size=batch_size,
                        num_workers=0,
                        verbose=False
                    )
                else:
                    raise e

        raw_segments = results[0] if results and len(results) > 0 else []

        parsed_segments: List[SpeakerSegment] = []
        for item in raw_segments:
            seg = self._parse_raw_segment(item)
            if seg is not None:
                parsed_segments.append(seg)

        processed_segments = self.postprocess_segments(
            parsed_segments,
            collar=collar,
            min_duration=min_duration
        )

        elapsed = time.time() - t0
        if verbose:
            print(f"[StreamingSortformer] Hoàn thành trong {elapsed:.2f}s | Tìm thấy {len(processed_segments)} phân đoạn phát biểu.")
        return processed_segments

    def diarize_numpy(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        batch_size: int = 1,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
        verbose: bool = False,
    ) -> List[SpeakerSegment]:
        """
        Thực hiện diarization trực tiếp trên mảng NumPy float32 trong bộ nhớ (không qua file tạm).
        Tối ưu cho luồng xử lý thời gian thực (Streaming Real-time).
        """
        if audio_pcm is None or len(audio_pcm) == 0:
            return []

        # Chuẩn hóa 1D float32
        audio_pcm = np.squeeze(audio_pcm).astype(np.float32)
        if len(audio_pcm) < int(0.2 * sample_rate):
            return []

        with torch.no_grad():
            try:
                results = self.model.diarize(
                    audio=audio_pcm,
                    sample_rate=sample_rate,
                    batch_size=batch_size,
                    num_workers=0,
                    verbose=False
                )
            except Exception as e:
                if verbose:
                    print(f"[StreamingSortformer] Lỗi diarize_numpy: {e}")
                return []

        raw_segments = results[0] if results and len(results) > 0 else []
        parsed_segments: List[SpeakerSegment] = []
        for item in raw_segments:
            seg = self._parse_raw_segment(item)
            if seg is not None:
                parsed_segments.append(seg)

        return self.postprocess_segments(
            parsed_segments,
            collar=collar,
            min_duration=min_duration
        )

    @staticmethod
    def detect_overlapping_speech(segments: List[SpeakerSegment]) -> List[Tuple[float, float, List[str]]]:
        """
        Phát hiện các khoảng thời gian có từ 2 người trở lên nói đè lên nhau (Overlapping Speech).
        Trả về danh sách các tuple: (start_time, end_time, [list_of_speakers])
        """
        if len(segments) < 2:
            return []

        # Tạo danh sách các sự kiện (thời điểm, loại sự kiện, speaker)
        events = []
        for seg in segments:
            events.append((seg.start, "start", seg.speaker))
            events.append((seg.end, "end", seg.speaker))

        events.sort(key=lambda x: x[0])

        active_speakers = set()
        overlap_regions = []
        overlap_start = None

        for t, event_type, spk in events:
            if event_type == "start":
                active_speakers.add(spk)
                if len(active_speakers) >= 2 and overlap_start is None:
                    overlap_start = t
            elif event_type == "end":
                if spk in active_speakers:
                    if len(active_speakers) >= 2 and overlap_start is not None and t > overlap_start:
                        overlap_regions.append((overlap_start, t, list(active_speakers)))
                    active_speakers.remove(spk)
                    if len(active_speakers) < 2:
                        overlap_start = None

        return overlap_regions

    @staticmethod
    def export_rttm(segments: List[SpeakerSegment], output_file: str, session_id: str = "audio") -> str:
        """
        Xuất danh sách phân đoạn ra file chuẩn RTTM.
        """
        out_dir = os.path.dirname(os.path.abspath(output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(seg.to_rttm_line(session_id=session_id) + "\n")
        print(f"[StreamingSortformer] Đã xuất file RTTM: '{output_file}'")
        return output_file

    @staticmethod
    def export_json(segments: List[SpeakerSegment], output_file: str) -> str:
        """
        Xuất danh sách phân đoạn ra file JSON chi tiết.
        """
        out_dir = os.path.dirname(os.path.abspath(output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        data = [seg.to_dict() for seg in segments]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[StreamingSortformer] Đã xuất file JSON: '{output_file}'")
        return output_file

    @staticmethod
    def print_timeline(segments: List[SpeakerSegment], overlaps: Optional[List] = None):
        """
        In bảng timeline phân tách người nói trực quan ra màn hình.
        """
        print("\n" + "=" * 70)
        print(f"{'THỜI GIAN (GIÂY)':<24} | {'THỜI LƯỢNG':<12} | {'NGƯỜI NÓI':<15} | {'ĐỘ TIN CẬY'}")
        print("-" * 70)

        for seg in segments:
            start_str = f"{seg.start:6.2f}s"
            end_str = f"{seg.end:6.2f}s"
            time_range = f"[{start_str} -> {end_str}]"
            dur_str = f"{seg.duration:5.2f}s"
            print(f"{time_range:<24} | {dur_str:<12} | {seg.speaker:<15} | {seg.confidence:.2%}")

        if overlaps:
            print("-" * 70)
            print(f"⚠️  PHÁT HIỆN {len(overlaps)} ĐOẠN NÓI ĐÈ LÊN NHAU (OVERLAPPING SPEECH):")
            for o_start, o_end, spk_list in overlaps:
                print(f"  • [{o_start:.2f}s -> {o_end:.2f}s] Các người nói: {', '.join(spk_list)}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Streaming Sortformer - Phân tách người nói theo thời gian thực"
    )
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn file âm thanh đầu vào (.wav)")
    parser.add_argument("--model", type=str, default="nvidia/diar_streaming_sortformer_4spk-v2.1", help="Tên checkpoint model NeMo")
    parser.add_argument("--device", type=str, default=None, help="'cuda' hoặc 'cpu'")
    parser.add_argument("--collar", type=float, default=0.25, help="Khoảng cách tối đa (s) để nối các đoạn cùng speaker")
    parser.add_argument("--min-dur", type=float, default=0.20, help="Thời lượng tối thiểu (s) của đoạn nói")
    parser.add_argument("--output-rttm", type=str, default=None, help="Đường dẫn lưu file RTTM đầu ra")
    parser.add_argument("--output-json", type=str, default=None, help="Đường dẫn lưu file JSON đầu ra")

    args = parser.parse_args()

    # Khởi tạo mô hình
    diarizer = StreamingSortformerDiarizer(
        model_name=args.model,
        device=args.device,
        collar=args.collar,
        min_duration=args.min_dur
    )

    # Chạy phân tách
    segments = diarizer.diarize_file(args.audio)
    overlaps = diarizer.detect_overlapping_speech(segments)

    # In kết quả trực quan
    diarizer.print_timeline(segments, overlaps)

    # Xuất file nếu yêu cầu
    if args.output_rttm:
        diarizer.export_rttm(segments, args.output_rttm)
    if args.output_json:
        diarizer.export_json(segments, args.output_json)


if __name__ == "__main__":
    main()
