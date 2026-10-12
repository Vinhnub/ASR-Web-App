# -*- coding: utf-8 -*-
"""
FastAPI Server & WebSocket Engine cho Realtime ASR Pipeline.
Hỗ trợ cả Streaming trực tiếp qua WebSocket và Upload file qua REST API.
Tích hợp giao diện điều khiển trực quan với công tắc Bật/Tắt từng module.
"""

import os
import sys
import json
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any, List

# Thiết lập UTF-8 cho Windows console để tránh UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    
    # Sửa lỗi ConnectionResetError (WinError 10054) khi client ngắt WebSocket đột ngột
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
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

import queue
import threading
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Query, HTTPException, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Đảm bảo import được các module cục bộ
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from config import PipelineConfig
from pipeline_manager import RealtimePipelineManager
from media_utils import extract_audio_from_media, generate_srt, generate_vtt, generate_txt

app = FastAPI(
    title="Realtime ASR Pipeline API",
    description="Hệ thống nhận dạng giọng nói & phân tách người nói thời gian thực với các module tùy biến linh hoạt.",
    version="1.0.0"
)

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn thư mục static files
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Khởi tạo Pipeline Manager
config = PipelineConfig()
pipeline_manager = RealtimePipelineManager(config=config)

# Bộ theo dõi các tác vụ xử lý offline đang chạy để hỗ trợ hủy tức thì
active_cancellation_events: Dict[str, threading.Event] = {}


@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """Phục vụ giao diện Web Client tương tác với chống cache."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers=headers)
    return HTMLResponse("<h1>Realtime ASR Pipeline Server đang hoạt động!</h1>", headers=headers)


# ==============================================================================
# 0. REST API: QUẢN LÝ & TRUY XUẤT FILE GHI ÂM MICROPHONE ONLINE
# ==============================================================================

RECORDINGS_DIR = os.path.join(STATIC_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

@app.api_route("/api/recording/latest", methods=["GET", "HEAD"])
async def get_latest_recording():
    """Tải hoặc nghe trực tiếp file âm thanh microphone đã thu gần nhất."""
    rec_file = os.path.join(RECORDINGS_DIR, "latest_recording.wav")
    if os.path.exists(rec_file):
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        return FileResponse(rec_file, media_type="audio/wav", filename="recording_online.wav", headers=headers)
    raise HTTPException(status_code=404, detail="Chưa có bản ghi âm microphone nào.")


@app.delete("/api/recording/latest")
async def delete_latest_recording():
    """Xóa file ghi âm cũ khi bắt đầu một phiên xử lý mới."""
    rec_file = os.path.join(RECORDINGS_DIR, "latest_recording.wav")
    if os.path.exists(rec_file):
        try:
            os.remove(rec_file)
            return {"status": "success", "message": "Đã xóa bản ghi âm microphone cũ."}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    return {"status": "success", "message": "Không có file nào cần xóa."}


@app.api_route("/api/recording/latest-offline", methods=["GET", "HEAD"])
async def get_latest_offline_recording():
    """Tải hoặc nghe trực tiếp file âm thanh đã xử lý/trích xuất gần nhất từ file offline."""
    rec_file = os.path.join(RECORDINGS_DIR, "latest_offline.wav")
    if os.path.exists(rec_file):
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        return FileResponse(rec_file, media_type="audio/wav", filename="recording_offline.wav", headers=headers)
    raise HTTPException(status_code=404, detail="Chưa có bản ghi âm offline nào.")


# ==============================================================================
# 1. WEBSOCKET STREAMING ENDPOINT (Microphone / Live Stream)
# ==============================================================================

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket nhận âm thanh thời gian thực từ Microphone hoặc Streaming Client.
    Định dạng gửi lên:
    - Text JSON: {"action": "configure", "options": {...}} hoặc {"action": "finish"}
    - Binary PCM: 16-bit PCM Mono 16000Hz.
    """
    await websocket.accept()
    session = pipeline_manager.create_session()
    session_id = session.session_id

    print(f"[WebSocket] Client đã kết nối -> Khởi tạo Session ID: {session_id}")

    try:
        # Gửi thông điệp chào mừng và cấu hình hiện tại
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "options": session.options,
            "message": "Kết nối thành công. Sẵn sàng nhận dữ liệu âm thanh PCM 16kHz!"
        }, ensure_ascii=False))

        while True:
            message = await websocket.receive()

            # 1. Nhận gói tin JSON cấu hình hoặc điều khiển
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action", "")

                    if action == "configure":
                        # Cập nhật các tùy chọn Bật / Tắt cho session hiện tại
                        new_opts = payload.get("options", {})
                        session.options.update(new_opts)
                        print(f"[WebSocket {session_id}] Đã cập nhật tùy chọn: {session.options}")
                        await websocket.send_text(json.dumps({
                            "type": "configured",
                            "options": session.options
                        }))

                    elif action == "start":
                        # Làm sạch và khởi tạo session mới tinh cho phiên thu âm mới
                        pipeline_manager.close_session(session_id)
                        prev_opts = session.options if ('session' in locals() and session) else {}
                        session = pipeline_manager.create_session(session_id=session_id, options_override=prev_opts)
                        await websocket.send_text(json.dumps({"type": "started", "session_id": session_id}))

                    elif action == "finish":
                        # Kích hoạt chốt câu và xuất kết quả cuối cùng
                        final_res = pipeline_manager.finalize_session(session_id)
                        await websocket.send_text(json.dumps(final_res, ensure_ascii=False))
                        # Tạo lại session mới để tiếp tục nhận lượt nói sau
                        session = pipeline_manager.create_session(session_id=session_id, options_override=session.options)

                    elif action == "reset":
                        pipeline_manager.close_session(session_id)
                        session = pipeline_manager.create_session(session_id=session_id, options_override=session.options)
                        await websocket.send_text(json.dumps({"type": "reset_done"}))

                except Exception as e:
                    # Đảm bảo session luôn tồn tại nếu gặp lỗi để không làm hỏng phiên kế tiếp
                    if not pipeline_manager.get_session(session_id):
                        try:
                            session = pipeline_manager.create_session(session_id=session_id)
                        except Exception:
                            pass
                    await websocket.send_text(json.dumps({"type": "error", "detail": str(e)}))

            # 2. Nhận gói tin nhị phân âm thanh PCM (Int16)
            elif "bytes" in message:
                raw_bytes = message["bytes"]
                if len(raw_bytes) > 0:
                    # Chuyển đổi Int16 -> Float32 [-1.0, 1.0]
                    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

                    # Đẩy qua Pipeline
                    partial_res = pipeline_manager.process_streaming_chunk(
                        session_id=session_id,
                        chunk_pcm=pcm_float32,
                        sample_rate=16000
                    )

                    # Gửi kết quả partial text về client
                    await websocket.send_text(json.dumps(partial_res, ensure_ascii=False))

    except WebSocketDisconnect:
        print(f"[WebSocket] Client ngắt kết nối session: {session_id}")
    except Exception as e:
        print(f"[WebSocket] Lỗi phiên stream {session_id}: {e}")
    finally:
        pipeline_manager.close_session(session_id)


# ==============================================================================
# 2. REST API: XỬ LÝ FILE ÂM THANH (WAV / MP3 / FLAC / M4A)
# ==============================================================================

# ==============================================================================
# 2. REST API: XỬ LÝ FILE VIDEO & ÂM THANH OFFLINE (MP4, MKV, AVI, WAV, MP3,...)
# ==============================================================================

@app.post("/api/process-file")
async def process_file_endpoint(
    file: UploadFile = File(...),
    denoise: bool = Form(True),
    asr: bool = Form(True),
    diarization: bool = Form(True),
    verification: bool = Form(True),
    hotwords: bool = Form(True),
    punctuation: bool = Form(True),
    summary: bool = Form(True),
    verification_threshold: Optional[float] = Form(None),
    min_speech_duration_for_id: Optional[float] = Form(None),
    diarization_collar: Optional[float] = Form(None),
    diarization_min_duration: Optional[float] = Form(None),
    asr_hotwords_score: Optional[float] = Form(None),
):
    """
    Tải lên file video hoặc âm thanh và xử lý trọn gói (đồng bộ).
    """
    options = {
        "denoise": denoise,
        "asr": asr,
        "diarization": diarization,
        "verification": verification,
        "hotwords": hotwords,
        "punctuation": punctuation,
        "summary": summary,
    }
    for param_k, param_v in [
        ("verification_threshold", verification_threshold),
        ("min_speech_duration_for_id", min_speech_duration_for_id),
        ("diarization_collar", diarization_collar),
        ("diarization_min_duration", diarization_min_duration),
        ("asr_hotwords_score", asr_hotwords_score),
    ]:
        if param_v is not None:
            options[param_k] = param_v

    temp_suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_file = tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False)
    temp_path = temp_file.name

    try:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()

        # Trích xuất âm thanh từ Video hoặc Audio
        audio_data, sr = extract_audio_from_media(temp_path, target_sr=16000)

        result = pipeline_manager.process_full_audio(
            audio_pcm=audio_data,
            sample_rate=sr,
            options_override=options
        )
        result["filename"] = file.filename
        result["source"] = "offline"
        return JSONResponse(result)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(err)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/process-file-stream")
async def process_file_stream_endpoint(
    request: Request,
    file: UploadFile = File(...),
    task_id: Optional[str] = Form(None),
    denoise: bool = Form(True),
    asr: bool = Form(True),
    diarization: bool = Form(True),
    verification: bool = Form(True),
    hotwords: bool = Form(True),
    punctuation: bool = Form(True),
    summary: bool = Form(True),
    verification_threshold: Optional[float] = Form(None),
    min_speech_duration_for_id: Optional[float] = Form(None),
    diarization_collar: Optional[float] = Form(None),
    diarization_min_duration: Optional[float] = Form(None),
    asr_hotwords_score: Optional[float] = Form(None),
):
    """
    Xử lý file video/audio offline với Server-Sent Events (SSE) báo cáo tiến trình thời gian thực.
    Hỗ trợ hủy tác vụ ngay lập tức qua API hoặc khi client ngắt kết nối.
    """
    if not task_id:
        task_id = str(uuid.uuid4())

    cancel_event = threading.Event()
    active_cancellation_events[task_id] = cancel_event

    options = {
        "denoise": denoise,
        "asr": asr,
        "diarization": diarization,
        "verification": verification,
        "hotwords": hotwords,
        "punctuation": punctuation,
        "summary": summary,
    }
    for param_k, param_v in [
        ("verification_threshold", verification_threshold),
        ("min_speech_duration_for_id", min_speech_duration_for_id),
        ("diarization_collar", diarization_collar),
        ("diarization_min_duration", diarization_min_duration),
        ("asr_hotwords_score", asr_hotwords_score),
    ]:
        if param_v is not None:
            options[param_k] = param_v

    temp_suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_file = tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False)
    temp_path = temp_file.name
    contents = await file.read()
    temp_file.write(contents)
    temp_file.close()

    filename = file.filename

    async def event_generator():
        q = queue.Queue(maxsize=100)

        def worker():
            try:
                if cancel_event.is_set():
                    raise InterruptedError("Quá trình xử lý đã bị hủy.")

                q.put({
                    "type": "progress",
                    "step": 1,
                    "total_steps": 7,
                    "step_name": "extract",
                    "percent": 5,
                    "message": f"Đang trích xuất âm thanh từ file '{filename}'..."
                })

                # Trích xuất âm thanh từ video / audio
                audio_data, sr = extract_audio_from_media(temp_path, target_sr=16000)
                dur = len(audio_data) / sr if sr > 0 else 0

                if cancel_event.is_set():
                    raise InterruptedError("Quá trình xử lý đã bị hủy.")

                q.put({
                    "type": "progress",
                    "step": 1,
                    "total_steps": 7,
                    "step_name": "extract",
                    "percent": 10,
                    "message": f"Trích xuất thành công: {dur:.1f}s âm thanh chuẩn 16kHz mono.",
                    "duration_s": dur
                })

                def on_progress(evt):
                    q.put({"type": "progress", **evt})

                result = pipeline_manager.process_full_audio_with_progress(
                    audio_pcm=audio_data,
                    sample_rate=sr,
                    options_override=options,
                    progress_callback=on_progress,
                    cancel_check=lambda: cancel_event.is_set()
                )
                result["filename"] = filename
                result["source"] = "offline"

                q.put({
                    "type": "complete",
                    "percent": 100,
                    "message": "Hoàn tất toàn bộ chu trình xử lý!",
                    "result": result
                })
            except InterruptedError as ie:
                q.put({
                    "type": "cancelled",
                    "message": str(ie) or "Quá trình xử lý đã được hủy thành công."
                })
            except Exception as e:
                q.put({
                    "type": "error",
                    "message": f"Lỗi trong quá trình xử lý: {str(e)}"
                })
            finally:
                q.put(None)  # Kết thúc stream
                active_cancellation_events.pop(task_id, None)
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Báo sự kiện khởi tạo kèm task_id để client biết
        yield f"data: {json.dumps({'type': 'init', 'task_id': task_id}, ensure_ascii=False)}\n\n"

        while True:
            # Phát hiện nếu client đóng kết nối (browser đóng, abort fetch)
            if await request.is_disconnected():
                cancel_event.set()
                active_cancellation_events.pop(task_id, None)
                break

            try:
                item = q.get_nowait()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("type") in ("complete", "error", "cancelled"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/cancel-process")
async def cancel_process_endpoint(
    task_id: Optional[str] = Query(None),
    payload: Optional[Dict[str, Any]] = Body(None)
):
    """
    Hủy tiến trình xử lý offline đang chạy theo task_id.
    """
    target_id = task_id or (payload.get("task_id") if payload else None)
    if not target_id:
        raise HTTPException(status_code=400, detail="Thiếu task_id để hủy.")

    ev = active_cancellation_events.get(target_id)
    if ev:
        ev.set()
        return {"status": "success", "message": f"Đã gửi tín hiệu hủy cho tác vụ '{target_id}'."}
    return {"status": "not_found", "message": "Không tìm thấy tác vụ đang xử lý hoặc tác vụ đã kết thúc."}



@app.post("/api/export-subtitles")
async def export_subtitles_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Xuất phụ đề SRT, WebVTT hoặc văn bản TXT từ danh sách phân đoạn.
    """
    export_format = payload.get("format", "srt").lower()
    segments = payload.get("segments", [])
    final_text = payload.get("final_text", "")
    filename = payload.get("filename", "transcript")
    base_name = os.path.splitext(filename)[0]

    if export_format == "srt":
        content = generate_srt(segments)
        media_type = "application/x-subrip"
        ext = ".srt"
    elif export_format == "vtt":
        content = generate_vtt(segments)
        media_type = "text/vtt"
        ext = ".vtt"
    else:
        content = generate_txt(segments, final_text=final_text)
        media_type = "text/plain; charset=utf-8"
        ext = ".txt"

    headers = {
        "Content-Disposition": f"attachment; filename=\"{base_name}{ext}\""
    }
    return Response(content=content, media_type=media_type, headers=headers)


@app.post("/api/summarize")
async def summarize_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Tạo bản tóm tắt nội dung trực tiếp theo yêu cầu từ văn bản hoặc danh sách phân đoạn.
    Sử dụng mô hình Qwen2.5-1.5B-Instruct.
    """
    text = payload.get("text", "")
    segments = payload.get("segments", [])
    max_tokens = payload.get("max_tokens")
    temperature = payload.get("temperature")

    if not text and not segments:
        raise HTTPException(status_code=400, detail="Cần cung cấp 'text' hoặc 'segments' để tóm tắt.")

    try:
        summary_result = pipeline_manager.summary.summarize(
            text=text,
            segments=segments,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return {"summary": summary_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tóm tắt: {str(e)}")


# ==============================================================================
# 3. REST API: QUẢN LÝ CƠ SỞ DỮ LIỆU NGƯỜI NÓI (TitaNet Enrollment)
# ==============================================================================

@app.get("/api/speakers")
def list_speakers():
    """Lấy danh sách người nói đã đăng ký."""
    return pipeline_manager.verification.list_enrolled_speakers()


@app.post("/api/speakers/enroll")
async def enroll_speaker(
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    """Đăng ký mẫu giọng nói mới (Voiceprint)."""
    temp_suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_file = tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False)
    temp_path = temp_file.name

    try:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()

        # Trích xuất âm thanh từ Video (MP4, MKV, AVI, MOV, WEBM) hoặc Audio (WAV, MP3, FLAC, M4A)
        # và chuẩn hóa về 16kHz Mono Float32
        audio_data, sr = extract_audio_from_media(temp_path, target_sr=16000)

        # Nếu module khử nhiễu đang bật, áp dụng khử nhiễu để voiceprint đồng nhất với pipeline
        if pipeline_manager.denoise.enabled and len(audio_data) > 0:
            try:
                audio_data = pipeline_manager.denoise.process(audio_data, sample_rate=sr)
            except Exception as denoise_err:
                print(f"[enroll_speaker] Cảnh báo khử nhiễu mẫu giọng: {denoise_err}")

        emb, count = pipeline_manager.verification.enroll(
            name=name,
            audio_pcm=audio_data,
            sample_rate=sr,
            description=description
        )
        return {
            "status": "success",
            "name": name,
            "sample_count": count,
            "message": f"Đã đăng ký thành công mẫu giọng nói cho '{name}'!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đăng ký giọng nói: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.delete("/api/speakers/{name}")
def delete_speaker(name: str):
    """Xóa một người nói khỏi Database."""
    success = pipeline_manager.verification.delete_speaker(name)
    if success:
        return {"status": "success", "message": f"Đã xóa người nói '{name}'."}
    raise HTTPException(status_code=404, detail=f"Không tìm thấy người nói '{name}'.")


# ==============================================================================
# 4. REST API: QUẢN LÝ TỪ KHÓA IN-DECODING HOTWORDS (ZipFormer Contextual Biasing)
# ==============================================================================

@app.get("/api/hotwords")
def get_hotwords():
    """Xem danh sách từ khóa in-decoding hotwords cho ZipFormer."""
    items = pipeline_manager.hotwords.get_hotwords_list()
    return {
        "hotwords": items,
        "count": len(items),
        "mode": "in-decoding",
        "default_score": pipeline_manager.config.asr_hotwords_score
    }


@app.post("/api/hotwords")
def add_hotword(phrase: str = Query(...), score: float = Query(2.0)):
    """Thêm hoặc cập nhật một từ khóa chuyên ngành vào ZipFormer Modified Beam Search."""
    clean_phrase = phrase.strip().upper()
    if not clean_phrase:
        raise HTTPException(status_code=400, detail="Từ khóa không được để trống.")
    pipeline_manager.hotwords.add_hotword(clean_phrase, score)
    pipeline_manager.asr.reload_hotwords(pipeline_manager.hotwords.hotwords_file)
    return {
        "status": "success",
        "message": f"Đã thêm từ khóa '{clean_phrase}' (Bonus Score: +{score}) vào ZipFormer!"
    }


@app.delete("/api/hotwords/{keyword}")
def remove_hotword(keyword: str):
    """Xóa một từ khóa khỏi ZipFormer."""
    clean_kw = keyword.strip().upper()
    success = pipeline_manager.hotwords.remove_hotword(clean_kw)
    if success:
        pipeline_manager.asr.reload_hotwords(pipeline_manager.hotwords.hotwords_file)
        return {"status": "success", "message": f"Đã xóa từ khóa '{clean_kw}' khỏi ZipFormer."}
    raise HTTPException(status_code=404, detail="Không tìm thấy từ khóa.")


# ==============================================================================
# 5. REST API: CẤU HÌNH THÔNG SỐ MODEL & CHỨC NĂNG (HYPERPARAMETERS)
# ==============================================================================

@app.get("/api/config")
def get_global_config():
    """Xem cấu hình và metadata thông số kỹ thuật của từng mô hình."""
    return pipeline_manager.get_model_parameters_meta()


@app.post("/api/config")
async def update_global_config(request: Request):
    """Cập nhật linh hoạt các thông số cho từng mô hình / chức năng."""
    try:
        payload = await request.json()
        updated = pipeline_manager.update_model_parameters(payload)
        return {
            "status": "success",
            "message": "Đã cập nhật các thông số model thành công!",
            "config": updated
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi cập nhật cấu hình: {str(e)}")


@app.post("/api/config/reset")
def reset_global_config():
    """Khôi phục toàn bộ thông số mô hình về giá trị mặc định tối ưu."""
    try:
        updated = pipeline_manager.reset_model_parameters()
        return {
            "status": "success",
            "message": "Đã khôi phục toàn bộ thông số mô hình về mặc định!",
            "config": updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khôi phục cấu hình: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\nKhởi động FastAPI server tại http://localhost:8000 ...")
    uvicorn.run("app:app", host=config.host, port=config.port, reload=False)
