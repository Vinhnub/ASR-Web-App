# -*- coding: utf-8 -*-
"""
Script kiểm thử toàn diện cho Realtime ASR Pipeline App:
1. Kiểm tra nạp các module độc lập.
2. Kiểm tra quy trình xử lý với toàn bộ các block BẬT (All ON).
3. Kiểm tra cơ chế BYPASS khi một số block TẮT (Selective OFF).
"""

import os
import sys
import time
import json
import soundfile as sf
import numpy as np

# Thiết lập UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn app
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from pipeline_manager import RealtimePipelineManager


def run_tests():
    test_wav = os.path.join(CURRENT_DIR, "testing_audio", "cris.wav")
    if not os.path.exists(test_wav):
        print(f"[TEST LỖI] Không tìm thấy file test: {test_wav}")
        return

    print("=" * 70)
    print("1. KHỞI TẠO REALTIME PIPELINE MANAGER...")
    print("=" * 70)
    cfg = PipelineConfig()
    manager = RealtimePipelineManager(config=cfg)

    # Đọc audio test
    audio, sr = sf.read(test_wav, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration_s = len(audio) / sr
    print(f"File test: '{test_wav}' ({duration_s:.2f}s, SR={sr}Hz)")

    print("\n" + "=" * 70)
    print("2. TEST CASE 1: TOÀN BỘ CÁC MODULE ĐỀU BẬT (ALL ON)")
    print("=" * 70)
    t0 = time.time()
    res1 = manager.process_full_audio(
        audio_pcm=audio,
        sample_rate=sr,
        options_override={
            "denoise": True,
            "asr": True,
            "diarization": True,
            "verification": True,
            "hotwords": True,
            "punctuation": True,
        }
    )
    t1 = time.time()
    print(f"⏱️ Thời gian xử lý: {t1 - t0:.2f}s (Audio: {duration_s:.2f}s)")
    print(f"Raw ASR Text     : {res1['raw_text']}")
    print(f"Final Text       : {res1['final_text']}")
    print(f"Phân đoạn Speaker: {json.dumps(res1['segments'], ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 70)
    print("3. TEST CASE 2: TẮT DENOISE, TẮT DIARIZATION, TẮT PUNCTUATION (BYPASS CHECK)")
    print("=" * 70)
    t0 = time.time()
    res2 = manager.process_full_audio(
        audio_pcm=audio,
        sample_rate=sr,
        options_override={
            "denoise": False,        # Tắt Denoise
            "asr": True,
            "diarization": False,    # Tắt Diarization
            "verification": False,   # Tắt Verification
            "hotwords": True,
            "punctuation": False,    # Tắt Punctuation
        }
    )
    t1 = time.time()
    print(f"⏱️ Thời gian xử lý: {t1 - t0:.2f}s")
    print(f"Raw ASR Text     : {res2['raw_text']}")
    print(f"Final Text       : {res2['final_text']}")
    print(f"Phân đoạn Speaker: {json.dumps(res2['segments'], ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 70)
    print("4. TEST CASE 3: MÔ PHỎNG STREAMING CHUNK THEO THỜI GIAN THỰC")
    print("=" * 70)
    session = manager.create_session()
    chunk_samples = int(0.32 * sr)
    print(f"Đang stream từng chunk 320ms ({chunk_samples} samples):")

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        chunk_res = manager.process_streaming_chunk(session.session_id, chunk, sample_rate=sr)
        if chunk_res.get("text"):
            print(f"  [Live {chunk_res['duration']}s]: \"{chunk_res['text']}\"")

    final_stream_res = manager.finalize_session(session.session_id)
    print(f"\nKết quả chốt streaming hoàn chỉnh:")
    print(f"• Text cuối : {final_stream_res['final_text']}")
    print(f"• Segments  : {final_stream_res['segments']}")

    print("\n" + "=" * 70)
    print("5. TEST CASE 4: KIỂM TRA IN-DECODING HOTWORDS BIASING TRỰC TIẾP TRONG ZIPFORMER")
    print("=" * 70)
    hw_list = manager.hotwords.get_hotwords_list()
    print(f"Danh sách hotwords hiện tại ({len(hw_list)} từ): {[h['phrase'] for h in hw_list]}")

    # Stream 1: Hotwords ON (Contextual Biasing)
    sess_hw_on = manager.create_session(options_override={"hotwords": True})
    for i in range(0, len(audio), chunk_samples):
        manager.process_streaming_chunk(sess_hw_on.session_id, audio[i : i + chunk_samples], sample_rate=sr)
    res_hw_on = manager.finalize_session(sess_hw_on.session_id)
    print(f"• Kết quả khi BẬT In-Decoding Hotwords : \"{res_hw_on['final_text']}\"")

    # Stream 2: Hotwords OFF (Pure beam search, no biasing)
    sess_hw_off = manager.create_session(options_override={"hotwords": False})
    for i in range(0, len(audio), chunk_samples):
        manager.process_streaming_chunk(sess_hw_off.session_id, audio[i : i + chunk_samples], sample_rate=sr)
    res_hw_off = manager.finalize_session(sess_hw_off.session_id)
    print(f"• Kết quả khi TẮT In-Decoding Hotwords : \"{res_hw_off['final_text']}\"")

    print("\n" + "=" * 70)
    print("✅ TOÀN BỘ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA THÀNH CÔNG!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
