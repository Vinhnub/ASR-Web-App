# -*- coding: utf-8 -*-
"""
================================================================================
Pipeline Tổng Hợp: Real-time Diarization + Speaker Voiceprint Identification
================================================================================
Kết hợp 2 mô hình Deep Learning hàng đầu của NVIDIA:
1. NVIDIA Streaming Sortformer: Phân đoạn người nói thời gian thực (spk_0, spk_1,...)
2. NVIDIA TitaNet: Trích xuất Voiceprint & Nhận diện danh tính thật từ kho hồ sơ giọng nói.

Đặc điểm nổi bật:
- Cơ chế "Khóa nhãn định danh" (Identity Lock): Trích xuất Voiceprint khi người nói
  đạt đủ độ dài tối thiểu (>= 0.8s) và lưu cache mapping, không tốn GPU tính lại mọi frame.
- Mô phỏng phát âm thanh theo thời gian thực (Real-time Pacing Simulation) để kiểm thử
  trải nghiệm luồng trực tiếp (Live Stream / Meeting Transcription).
- Tự động phát hiện và trích xuất mẫu âm thanh của người lạ (Auto-save Unknown Speakers)
  để thuận tiện cho việc nghe lại và đăng ký sau cuộc họp.
- Xuất báo cáo đầy đủ: File chuẩn RTTM, file chi tiết JSON, và tóm tắt biên bản cuộc họp.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

# Thiết lập UTF-8 cho Windows console
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

# Vá lỗi CPython TemporaryDirectory rmtree trên Windows khi manifest.json bị khóa
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


# Đảm bảo đường dẫn import tương đối từ thư mục cha nếu chạy từ script cục bộ
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from audio_utils import load_audio
except ImportError:
    # Hàm dự phòng nếu không tìm thấy audio_utils
    def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)

from streaming_sortformer import StreamingSortformerDiarizer, SpeakerSegment
from speaker_recognition_titanet import TitaNetSpeakerRecognizer, SpeakerDatabase


class StreamingDiarizationPipeline:
    """
    Pipeline tích hợp toàn diện: Phân tách lượt nói và Gán danh tính người thật theo thời gian thực.
    """

    def __init__(
        self,
        sortformer_model: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
        titanet_model: str = "nvidia/speakerverification_en_titanet_large",
        db_dir: str = "./speaker_db",
        threshold: float = 0.70,
        min_speech_duration_for_id: float = 0.80,
        device: Optional[str] = None,
        auto_save_unknown: bool = False,
    ):
        """
        Khởi tạo pipeline với cả 2 mô hình và cơ sở dữ liệu.

        Tham số:
            sortformer_model: Checkpoint Sortformer Diarization
            titanet_model: Checkpoint TitaNet Speaker Verification
            db_dir: Thư mục chứa cơ sở dữ liệu giọng nói đã đăng ký
            threshold: Ngưỡng Cosine Similarity để nhận diện người quen (0.0 -> 1.0)
            min_speech_duration_for_id: Thời lượng tiếng nói tối thiểu (giây) để kích hoạt nhận diện
            device: 'cuda' hoặc 'cpu' (tự động chọn nếu None)
            auto_save_unknown: Tự động lưu các đoạn audio của người chưa rõ danh tính ra thư mục
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("=" * 70)
        print("KHỞI TẠO PIPELINE: STREAMING SORTFORMER + TITANET IDENTIFICATION")
        print(f"• Thiết bị thực thi: {self.device.upper()}")
        print(f"• Ngưỡng nhận diện (Threshold): {threshold}")
        print("=" * 70)

        # 1. Khởi tạo mô hình Diarization (Sortformer)
        self.diarizer = StreamingSortformerDiarizer(
            model_name=sortformer_model,
            device=self.device
        )

        # 2. Khởi tạo mô hình Speaker Recognition (TitaNet)
        self.recognizer = TitaNetSpeakerRecognizer(
            model_name=titanet_model,
            device=self.device
        )

        # 3. Kết nối với cơ sở dữ liệu giọng nói
        self.db = SpeakerDatabase(
            db_dir=db_dir,
            recognizer=self.recognizer
        )

        self.threshold = threshold
        self.min_speech_duration_for_id = min_speech_duration_for_id
        self.auto_save_unknown = auto_save_unknown

    def _extract_speaker_audio_slices(
        self,
        audio_array: np.ndarray,
        segments: List[SpeakerSegment],
        sample_rate: int = 16000
    ) -> Dict[str, np.ndarray]:
        """
        Trích xuất và ghép các đoạn phát biểu của từng anonymous speaker (spk_0, spk_1,...)
        thành một mảng âm thanh liên tục phục vụ cho việc lấy Voiceprint có độ chính xác cao nhất.
        """
        speaker_audio_map: Dict[str, List[np.ndarray]] = {}

        total_samples = len(audio_array)

        for seg in segments:
            start_sample = int(seg.start * sample_rate)
            end_sample = min(total_samples, int(seg.end * sample_rate))

            if end_sample > start_sample:
                clip = audio_array[start_sample:end_sample]
                if seg.speaker not in speaker_audio_map:
                    speaker_audio_map[seg.speaker] = []
                speaker_audio_map[seg.speaker].append(clip)

        # Nối các mảng lại với nhau cho từng speaker
        concatenated_map: Dict[str, np.ndarray] = {}
        for spk, clips in speaker_audio_map.items():
            if clips:
                concatenated_map[spk] = np.concatenate(clips, axis=0)

        return concatenated_map

    def resolve_speaker_identities(
        self,
        audio_array: np.ndarray,
        raw_segments: List[SpeakerSegment],
        sample_rate: int = 16000,
        output_unknown_dir: Optional[str] = None,
    ) -> Dict[str, Tuple[str, float]]:
        """
        Thực hiện cơ chế "Khóa định danh" (Identity Lock):
        - Gom các đoạn audio của từng nhãn ẩn danh (spk_0, spk_1,...).
        - Trích xuất Voiceprint và so khớp với Database.
        - Trả về bản đồ ánh xạ: {raw_speaker: (real_name, similarity_score)}.
        """
        speaker_audio = self._extract_speaker_audio_slices(audio_array, raw_segments, sample_rate=sample_rate)
        identity_map: Dict[str, Tuple[str, float]] = {}

        print("\n[Pipeline] Đang phân tích Voiceprint & So khớp danh tính người nói:")

        for raw_spk, audio_clip in speaker_audio.items():
            duration_s = len(audio_clip) / sample_rate

            if duration_s < self.min_speech_duration_for_id:
                print(f"  • {raw_spk}: Thời lượng lời nói quá ngắn ({duration_s:.2f}s < {self.min_speech_duration_for_id}s) -> Giữ nguyên nhãn tạm.")
                identity_map[raw_spk] = (f"Chưa rõ ({raw_spk})", 0.0)
                continue

            # Trích xuất embedding bằng TitaNet
            emb = self.recognizer.extract_embedding(audio_clip, sample_rate=sample_rate)

            # So khớp với database
            matched_name, sim_score = self.db.identify(emb, threshold=self.threshold)

            if matched_name != "Unknown":
                print(f"  • {raw_spk}  ==>  [ĐÃ NHẬN DIỆN]: '{matched_name}' (Độ tương đồng: {sim_score:.2%})")
                identity_map[raw_spk] = (matched_name, sim_score)
            else:
                unknown_display = f"Người lạ ({raw_spk})"
                print(f"  • {raw_spk}  ==>  [CHƯA ĐĂNG KÝ]: Khách mới (Score cao nhất chỉ đạt: {sim_score:.2%})")
                identity_map[raw_spk] = (unknown_display, sim_score)

                # Lưu mẫu người lạ nếu bật tùy chọn
                if self.auto_save_unknown and output_unknown_dir:
                    os.makedirs(output_unknown_dir, exist_ok=True)
                    unknown_wav_path = os.path.join(output_unknown_dir, f"{raw_spk}_sample_{int(time.time())}.wav")
                    try:
                        import soundfile as sf
                        sf.write(unknown_wav_path, audio_clip, sample_rate)
                        print(f"    [Auto-Save] Đã lưu mẫu âm thanh của người lạ ra '{unknown_wav_path}'")
                    except Exception as e:
                        print(f"    [Auto-Save] Lỗi lưu mẫu: {e}")

        return identity_map

    def process_file(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        simulate_streaming: bool = False,
        pacing_interval_s: float = 0.5,
    ) -> List[Dict]:
        """
        Xử lý toàn bộ quy trình trên một file âm thanh.

        Tham số:
            audio_path: Đường dẫn file âm thanh đầu vào (.wav 16kHz)
            output_dir: Thư mục xuất các file báo cáo (.rttm, .json, .txt)
            simulate_streaming: Bật chế độ mô phỏng phát luồng thời gian thực
            pacing_interval_s: Bước nhảy thời gian khi mô phỏng

        Trả về:
            Danh sách các phân đoạn đã được gán danh tính người thật.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: '{audio_path}'")

        print(f"\n[Pipeline] Bắt đầu xử lý: '{audio_path}'")
        total_t0 = time.time()

        # 1. Tải và chuẩn hóa audio thành mono 16kHz float32
        audio_array = load_audio(audio_path, target_sr=16000)
        audio_duration_s = len(audio_array) / 16000.0
        print(f"[Pipeline] Thời lượng âm thanh: {audio_duration_s:.2f}s ({len(audio_array)} samples)")

        # 2. Chạy Streaming Sortformer Diarization
        raw_segments = self.diarizer.diarize_file(audio_path)
        overlaps = self.diarizer.detect_overlapping_speech(raw_segments)

        # 3. Phân tích Voiceprint và nhận diện danh tính bằng TitaNet
        unknown_dir = os.path.join(output_dir, "unknown_speakers") if output_dir else None
        identity_map = self.resolve_speaker_identities(
            audio_array,
            raw_segments,
            sample_rate=16000,
            output_unknown_dir=unknown_dir
        )

        # 4. Gán danh tính thật vào từng phân đoạn
        final_segments: List[Dict] = []
        for seg in raw_segments:
            real_name, score = identity_map.get(seg.speaker, (seg.speaker, 0.0))
            record = {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "duration": round(seg.duration, 3),
                "speaker": real_name,
                "raw_speaker": seg.speaker,
                "similarity_score": round(score, 4),
                "confidence": round(seg.confidence, 4)
            }
            final_segments.append(record)

        # 5. Nếu bật chế độ mô phỏng streaming
        if simulate_streaming:
            self._simulate_streaming_pacing(final_segments, audio_duration_s, pacing_interval_s)
        else:
            self._print_final_timeline(final_segments, overlaps)

        # 6. Xuất kết quả nếu có output_dir
        if output_dir:
            self._export_all_reports(audio_path, final_segments, output_dir)

        total_elapsed = time.time() - total_t0
        print(f"[Pipeline] Hoàn thành toàn bộ quy trình trong {total_elapsed:.2f}s (Tỉ lệ RTFX: {audio_duration_s / max(total_elapsed, 1e-4):.2f}x)")
        return final_segments

    def _simulate_streaming_pacing(
        self,
        segments: List[Dict],
        total_duration: float,
        interval: float = 0.5
    ):
        """
        Mô phỏng phát luồng thời gian thực ra màn hình console (như đang họp live).
        """
        print("\n" + "=" * 70)
        print(">>> BẮT ĐẦU MÔ PHỎNG PHÁT LUỒNG TRỰC TIẾP THEO THỜI GIAN THỰC <<<")
        print("=" * 70)

        current_time = 0.0
        active_idx = 0
        n_segs = len(segments)

        while current_time <= total_duration:
            # Kiểm tra xem có segment nào đang nói tại thời điểm này
            while active_idx < n_segs and segments[active_idx]["end"] < current_time:
                active_idx += 1

            current_active = []
            check_idx = active_idx
            while check_idx < n_segs and segments[check_idx]["start"] <= current_time:
                if segments[check_idx]["end"] >= current_time:
                    current_active.append(segments[check_idx]["speaker"])
                check_idx += 1

            mins = int(current_time // 60)
            secs = current_time % 60
            time_tag = f"[{mins:02d}:{secs:05.2f}]"

            if current_active:
                spk_str = " & ".join(current_active)
                print(f"{time_tag} 🎤 ĐANG NÓI: {spk_str}", flush=True)
            else:
                print(f"{time_tag} ... [Im lặng]", flush=True)

            time.sleep(interval)
            current_time += interval

        print("=" * 70)
        print(">>> KẾT THÚC MÔ PHỎNG LUỒNG TRỰC TIẾP <<<")
        print("=" * 70 + "\n")

    @staticmethod
    def _print_final_timeline(segments: List[Dict], overlaps: Optional[List] = None):
        """
        In bảng kết quả phân tách và định danh hoàn chỉnh ra màn hình.
        """
        print("\n" + "=" * 80)
        print(f"{'THỜI GIAN':<24} | {'THỜI LƯỢNG':<10} | {'DANH TÍNH ĐƯỢC NHẬN DIỆN':<26} | {'ĐỘ KHỚP'}")
        print("-" * 80)

        for seg in segments:
            time_range = f"[{seg['start']:6.2f}s -> {seg['end']:6.2f}s]"
            dur_str = f"{seg['duration']:5.2f}s"
            sim_str = f"{seg['similarity_score']:.1%}" if seg['similarity_score'] > 0 else "N/A"
            print(f"{time_range:<24} | {dur_str:<10} | {seg['speaker']:<26} | {sim_str}")

        if overlaps:
            print("-" * 80)
            print(f"⚠️  PHÁT HIỆN {len(overlaps)} VÙNG CÓ 2 NGƯỜI NÓI CHỒNG LÊN NHAU (OVERLAP):")
            for o_start, o_end, spks in overlaps:
                print(f"  • Từ {o_start:.2f}s đến {o_end:.2f}s: Giữa {', '.join(spks)}")

        print("=" * 80 + "\n")

    def _export_all_reports(self, audio_path: str, segments: List[Dict], output_dir: str):
        """
        Xuất toàn bộ các định dạng file kết quả: RTTM, JSON, và TXT Summary.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # 1. Xuất file RTTM
        rttm_path = os.path.join(output_dir, f"{base_name}.rttm")
        with open(rttm_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(
                    f"SPEAKER {base_name} 1 {seg['start']:.3f} {seg['duration']:.3f} "
                    f"<NA> <NA> {seg['speaker'].replace(' ', '_')} <NA> <NA>\n"
                )
        print(f"[Export] Đã xuất file RTTM: '{rttm_path}'")

        # 2. Xuất file JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"[Export] Đã xuất file JSON: '{json_path}'")

        # 3. Xuất file TXT biên bản tóm tắt
        txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        speaker_total_dur: Dict[str, float] = {}
        for s in segments:
            spk = s["speaker"]
            speaker_total_dur[spk] = speaker_total_dur.get(spk, 0.0) + s["duration"]

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"TÓM TẮT CUỘC HỌP / PHÂN ĐOẠN NGƯỜI NÓI: {base_name}\n")
            f.write(f"Ngày xử lý: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("THỐNG KÊ THỜI LƯỢNG NÓI THEO TỪNG THÀNH VIÊN:\n")
            for spk, dur in sorted(speaker_total_dur.items(), key=lambda x: -x[1]):
                f.write(f"  • {spk:<25}: {dur:6.2f} giây\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("TIMELINE CHI TIẾT TỪNG LƯỢT NÓI:\n")
            for s in segments:
                f.write(f"[{s['start']:6.2f}s -> {s['end']:6.2f}s] {s['speaker']}\n")
        print(f"[Export] Đã xuất file tóm tắt: '{txt_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Phân tách & Nhận diện Danh tính Người nói Thời gian thực (Sortformer + TitaNet)"
    )
    parser.add_argument("--audio", type=str, required=True, help="File âm thanh cần phân tích (.wav)")
    parser.add_argument("--db-dir", type=str, default="./speaker_db", help="Thư mục cơ sở dữ liệu hồ sơ giọng nói")
    parser.add_argument("--output-dir", type=str, default="./diarization_results", help="Thư mục lưu kết quả xuất")
    parser.add_argument("--threshold", type=float, default=0.70, help="Ngưỡng nhận diện Cosine Similarity")
    parser.add_argument("--min-speech-dur", type=float, default=0.80, help="Thời lượng nói tối thiểu để trích xuất Voiceprint")
    parser.add_argument("--simulate-streaming", action="store_true", help="Bật chế độ mô phỏng trực tiếp theo thời gian thực")
    parser.add_argument("--auto-save-unknown", action="store_true", help="Tự động lưu audio của người chưa rõ danh tính")
    parser.add_argument("--device", type=str, default=None, help="'cuda' hoặc 'cpu'")

    args = parser.parse_args()

    pipeline = StreamingDiarizationPipeline(
        db_dir=args.db_dir,
        threshold=args.threshold,
        min_speech_duration_for_id=args.min_speech_dur,
        device=args.device,
        auto_save_unknown=args.auto_save_unknown
    )

    pipeline.process_file(
        audio_path=args.audio,
        output_dir=args.output_dir,
        simulate_streaming=args.simulate_streaming
    )


if __name__ == "__main__":
    main()
