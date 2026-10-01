import os
import sys
import re
import wave
import struct
import math
import numpy as np

# Thiết lập UTF-8 cho Windows console để tránh UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def create_dummy_wav(file_path: str = "test.wav", duration_s: float = 5.0, sample_rate: int = 16000):
    """
    Tạo một file WAV mẫu 16kHz mono nếu chưa có file audio nào để test.
    """
    if os.path.exists(file_path):
        return file_path
    
    print(f"[audio_utils] Khởi tạo file audio mẫu '{file_path}' ({duration_s}s, {sample_rate}Hz)...", flush=True)
    num_samples = int(duration_s * sample_rate)
    with wave.open(file_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = (0.5 * math.sin(2.0 * math.pi * 440.0 * t) + 
                     0.3 * math.sin(2.0 * math.pi * 880.0 * t) +
                     0.2 * math.sin(2.0 * math.pi * 1320.0 * t))
            envelope = math.exp(-3.0 * (t % 1.0))
            sample_val = int(value * envelope * 20000.0)
            sample_val = max(-32768, min(32767, sample_val))
            frames.extend(struct.pack("<h", sample_val))
            
        wav_file.writeframes(frames)
    
    print(f"[audio_utils] Đã tạo thành công '{file_path}'.", flush=True)
    return file_path


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Tải file âm thanh, đảm bảo chuyển đổi thành Mono và chuẩn hóa biên độ float32 [-1.0, 1.0].
    """
    if not os.path.exists(file_path):
        create_dummy_wav(file_path, duration_s=5.0, sample_rate=target_sr)
    
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype="float32")
    except ImportError:
        with wave.open(file_path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            channels = wf.getnchannels()
            raw_bytes = wf.readframes(n_frames)
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            num_target_samples = int(len(audio) * target_sr / sr)
            audio = np.interp(
                np.linspace(0, len(audio), num_target_samples, endpoint=False),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)
            
    return audio.astype(np.float32)


def slice_sliding_windows(
    audio: np.ndarray, 
    sample_rate: int = 16000, 
    window_s: float = 2.0, 
    slide_s: float = 0.5
):
    """
    Chia mảng âm thanh thành các cửa sổ con (windows) window_s, bước trượt slide_s.
    """
    if window_s <= 0:
        raise ValueError(f"Lỗi: window_s (độ dài cửa sổ) phải > 0! (Hiện tại: {window_s})")
    if slide_s <= 0:
        raise ValueError(
            f"Lỗi: slide_s (bước trượt) phải > 0! (Hiện tại: {slide_s})\n"
            f"💡 Gợi ý:\n"
            f"   - Nếu bạn muốn KHÔNG bị trùng lặp (Overlap = 0s), hãy đặt: slide_s = {window_s} (bằng đúng window_s).\n"
            f"   - Nếu bạn muốn gối đầu: hãy đặt 0 < slide_s < {window_s} (ví dụ: slide_s = 0.5s)."
        )

    window_samples = int(window_s * sample_rate)
    slide_samples = int(slide_s * sample_rate)
    num_samples = len(audio)
    
    chunks = []
    if num_samples <= window_samples:
        padded = np.pad(audio, (0, window_samples - num_samples), mode="constant")
        chunks.append({
            "index": 0,
            "start_s": 0.0,
            "end_s": round(window_s, 2),
            "audio": padded
        })
        return chunks
        
    start = 0
    idx = 0
    while True:
        end = start + window_samples
        if end <= num_samples:
            chunk = audio[start:end]
        else:
            chunk = audio[start:]
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")
            
        start_s = round(start / sample_rate, 2)
        end_s = round((start + window_samples) / sample_rate, 2)
        
        chunks.append({
            "index": idx,
            "start_s": start_s,
            "end_s": end_s,
            "audio": chunk
        })
        
        idx += 1
        if end >= num_samples:
            break
        start += slide_samples
            
    return chunks


def _clean_word(w: str) -> str:
    """Xóa ký tự đặc biệt và đưa về chữ thường để so sánh."""
    return re.sub(r"[^\w\s]", "", w.lower()).strip()


def merge_sliding_texts(full_text: str, new_chunk_text: str, max_lookback_words: int = 15):
    """
    Thuật toán khử trùng lặp và ghép văn bản gối đầu cơ bản.
    Trả về tuple 3 giá trị:
        (updated_full_text, newly_added_words, overlap_words_removed)
    """
    new_chunk_text = new_chunk_text.strip()
    if not full_text:
        return new_chunk_text, new_chunk_text, ""
    if not new_chunk_text:
        return full_text.strip(), "", ""

    words_full = full_text.strip().split()
    words_new = new_chunk_text.split()

    clean_full = [_clean_word(w) for w in words_full]
    clean_new = [_clean_word(w) for w in words_new]

    # Quét so khớp từ lớn nhất đến nhỏ nhất
    max_check = min(len(clean_full), len(clean_new), max_lookback_words)
    for k in range(max_check, 0, -1):
        if clean_full[-k:] == clean_new[:k]:
            overlap_str = " ".join(words_new[:k])
            if k < len(words_new):
                added_str = " ".join(words_new[k:])
                return full_text + " " + added_str, added_str, overlap_str
            else:
                return full_text, "", overlap_str

    # Kiểm tra nếu chunk mới đã nằm trọn trong đoạn gần nhất
    recent_full_str = " ".join(clean_full[-max_lookback_words:])
    new_str = " ".join(clean_new)
    if new_str and new_str in recent_full_str:
        return full_text, "", new_chunk_text

    return full_text + " " + new_chunk_text, new_chunk_text, ""


def display_window_progress(
    item: dict, 
    latency: float, 
    window_s: float, 
    raw_text: str, 
    added_text: str = "", 
    overlap_text: str = "", 
    current_full_text: str = ""
):
    """
    Hiển thị chi tiết thời gian thực quá trình xử lý của từng Window.
    """
    idx = item["index"] + 1
    t_range = f"{item['start_s']:05.2f}s -> {item['end_s']:05.2f}s"
    rtf = latency / window_s
    lat_ms = latency * 1000

    print(f"\n┌── [Window #{idx:02d}] [{t_range}]  ⏱️ Thời gian: {lat_ms:6.2f} ms | RTF: {rtf:.3f}x", flush=True)
    print(f"│  • Nhận dạng 2s gốc  : \"{raw_text}\"", flush=True)
    if overlap_text:
        print(f"│  • Đoạn trùng (lọc)  : \"{overlap_text}\"", flush=True)
    else:
        print(f"│  • Đoạn trùng (lọc)  : (không có)", flush=True)
    if added_text:
        print(f"│  • Từ mới ghép thêm  : + \"{added_text}\"", flush=True)
    else:
        print(f"│  • Từ mới ghép thêm  : (không có từ mới)", flush=True)
    print(f"└── 💬 Câu tích lũy    : \"{current_full_text}\"", flush=True)


def print_benchmark_summary(model_name: str, latencies: list, window_s: float = 2.0, merged_text: str = ""):
    """
    In bảng tổng kết đo thời gian xử lý và văn bản sau khi khử trùng lặp.
    """
    if not latencies:
        return
        
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    rtf = avg_latency / window_s
    speedup = 1.0 / rtf if rtf > 0 else float("inf")
    
    print("\n" + "=" * 70, flush=True)
    print(f"📊 BÁO CÁO HIỆU NĂNG - {model_name.upper()}", flush=True)
    print("=" * 70, flush=True)
    print(f"• Kích thước window        : {window_s:.1f}s ({int(window_s * 16000)} samples)", flush=True)
    print(f"• Số lượng samples đã chạy : {len(latencies)}", flush=True)
    print(f"• Thời gian xử lý 1 sample :", flush=True)
    print(f"    - Trung bình (Avg)     : {avg_latency * 1000:.2f} ms ({avg_latency:.4f}s)", flush=True)
    print(f"    - Nhanh nhất (Min)     : {min_latency * 1000:.2f} ms", flush=True)
    print(f"    - Lâu nhất (Max)       : {max_latency * 1000:.2f} ms", flush=True)
    print(f"• Real-Time Factor (RTF)   : {rtf:.4f} " + ("(⚡ Nhanh hơn realtime)" if rtf < 1.0 else "(⚠️ Chậm hơn realtime)"), flush=True)
    print(f"• Tốc độ xử lý             : {speedup:.1f}x realtime", flush=True)
    if merged_text:
        print("-" * 70, flush=True)
        print("📝 VĂN BẢN HOÀN CHỈNH TOÀN BÀI (ĐÃ KHỬ TRÙNG LẶP):", flush=True)
        print(f"   \"{merged_text.strip()}\"", flush=True)
    print("=" * 70 + "\n", flush=True)


def save_benchmark_result(
    output_file: str,
    model_name: str,
    audio_path: str,
    latencies: list,
    window_s: float = 2.0,
    slide_s: float = 0.5,
    merged_text: str = "",
    window_records: list = None
):
    """
    Lưu kết quả văn bản cuối cùng và báo cáo hiệu năng vào file .txt UTF-8.
    """
    if not latencies:
        return

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    rtf = avg_latency / window_s
    speedup = 1.0 / rtf if rtf > 0 else float("inf")

    lines = []
    lines.append("=" * 75)
    lines.append(f"BÁO CÁO KẾT QUẢ NHẬN DẠNG GIỌNG NÓI (ASR)")
    lines.append("=" * 75)
    lines.append(f"• Mô hình              : {model_name}")
    lines.append(f"• File âm thanh        : {audio_path}")
    lines.append(f"• Cấu hình Window      : Window = {window_s:.1f}s | Slide = {slide_s:.1f}s | Overlap = {window_s - slide_s:.1f}s")
    lines.append(f"• Tổng số samples      : {len(latencies)}")
    lines.append("-" * 75)
    lines.append("THỐNG KÊ HIỆU NĂNG XỬ LÝ:")
    lines.append(f"• Thời gian xử lý 1 sample (2.0s):")
    lines.append(f"    - Trung bình (Avg) : {avg_latency * 1000:.2f} ms ({avg_latency:.4f}s)")
    lines.append(f"    - Nhanh nhất (Min) : {min_latency * 1000:.2f} ms")
    lines.append(f"    - Lâu nhất (Max)   : {max_latency * 1000:.2f} ms")
    lines.append(f"• Real-Time Factor (RTF): {rtf:.4f} (" + ("Nhanh hơn realtime" if rtf < 1.0 else "Chậm hơn realtime") + ")")
    lines.append(f"• Tốc độ so với realtime: {speedup:.1f}x")
    lines.append("-" * 75)
    lines.append("VĂN BẢN HOÀN CHỈNH CUỐI CÙNG (ĐÃ KHỬ TRÙNG LẶP):")
    lines.append(f"{merged_text.strip() if merged_text else '(Không có kết quả)'}")
    lines.append("-" * 75)

    if window_records:
        lines.append("CHI TIẾT TỪNG CỬA SỔ (WINDOW):")
        for rec in window_records:
            idx = rec.get("index", 0) + 1
            t_range = f"{rec.get('start_s', 0):05.2f}s -> {rec.get('end_s', 0):05.2f}s"
            lat_ms = rec.get("latency", 0) * 1000
            raw = rec.get("raw_text", "")
            added = rec.get("added_text", "")
            lines.append(f"  [Window #{idx:02d}] [{t_range}] ({lat_ms:6.2f} ms): raw=\"{raw}\" | +=\"{added}\"")
        lines.append("=" * 75)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"💾 [Lưu file] Đã lưu kết quả kiểm tra vào file: '{output_file}'", flush=True)
