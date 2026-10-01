# Realtime ASR Pipeline WebApp 🎙️

Hệ thống nhận dạng giọng nói Tiếng Việt (Speech-to-Text) theo thời gian thực được tối ưu hóa cho môi trường độ trễ thấp. Ứng dụng là một bản "đóng gói" (self-contained) hoàn chỉnh, kết hợp sức mạnh của ONNX Runtime và PyTorch để cân bằng giữa hiệu năng và tính ổn định.

## 🚀 Các Tính Năng Cốt Lõi (Modules)
1. **Denoise (Lọc nhiễu)**: Sử dụng DeepFilterNet3 (PyTorch) cho chất lượng lọc tiếng ồn theo thời gian thực cực kỳ xuất sắc.
2. **ASR (Nhận dạng giọng nói)**: Sử dụng mô hình Zipformer qua thư viện `sherpa-onnx` (C++ backend) siêu nhẹ và nhanh.
3. **Diarization (Phân tách người nói)**: Sử dụng Streaming SortFormer (NVIDIA NeMo - PyTorch) để nhận diện thời gian nói của từng người.
4. **Speaker Verification (Xác thực người nói)**: Sử dụng TitaNet (ONNX Runtime) để lưu trữ và so sánh sinh trắc học giọng nói.
5. **Hotwords (Từ khóa ưu tiên)**: Tích hợp Contextual Biasing ngay trong quá trình giải mã (In-decoding) để bắt chính xác các danh từ riêng, thuật ngữ kỹ thuật.
6. **Punctuation (Điền dấu câu)**: Sử dụng ViBERT-capu (ONNX) để khôi phục dấu câu tự nhiên cho văn bản.

---

## 📂 Cấu Trúc Thư Mục
Dự án được thiết kế độc lập hoàn toàn, tất cả mọi thứ cần thiết để chạy đều nằm gọn trong thư mục này:
```text
ASR_Webapp/
│
├── app.py                      # File khởi chạy FastAPI server (Backend & WebSocket)
├── config.py                   # Cấu hình trung tâm (Đường dẫn, các thông số pipeline)
├── pipeline_manager.py         # Trình quản lý điều phối dữ liệu qua các modules
├── test_pipeline.py            # Script kiểm thử hệ thống tự động
├── hotwords.txt                # Danh sách từ khóa ưu tiên (Hotwords)
│
├── modules/                    # Chứa mã nguồn liên kết từng block (ASR, Denoise, ONNX wrappers,...)
├── models/                     # Chứa các file mô hình ONNX (như titanet_large.onnx)
├── static/                     # Giao diện Web Client (HTML, JS, CSS)
├── speaker_db/                 # Cây dữ liệu lưu trữ vector nhúng của giọng nói người dùng
│
├── NoiseFilter/                # Mã nguồn & mô hình Denoise (DeepFilterNet)
├── Speaker_Diarization/        # Mã nguồn Diarization (Streaming SortFormer)
├── zipformer_streaming_vi/     # Mô hình acoustic ASR Zipformer
├── Automatic_Punctuation/      # Mã nguồn và mô hình điền dấu câu
└── testing_audio/              # Các file âm thanh mẫu dùng để test
```

---

## ⚙️ Yêu Cầu Hệ Thống & Cài Đặt
Ứng dụng yêu cầu môi trường Python chứa các thư viện AI chuyên sâu. 
Giả định bạn đang sử dụng môi trường Conda đã được thiết lập sẵn (ví dụ: `ai_gpu`).

Các thư viện chính bao gồm:
- `torch`, `torchaudio`
- `nemo_toolkit` (NVIDIA NeMo)
- `onnxruntime` (hoặc `onnxruntime-gpu` nếu dùng card rời)
- `sherpa-onnx`
- `fastapi`, `uvicorn`, `websockets`

---

## 💻 Hướng Dẫn Chạy Ứng Dụng

### 1. Kích hoạt môi trường
Mở Terminal / Anaconda Prompt và kích hoạt môi trường của bạn:
```bash
conda activate ai_gpu
```

### 2. Khởi chạy Web Server (Giao diện Web)
Đảm bảo bạn đang đứng ở thư mục `ASR_Webapp`, sau đó chạy:
```bash
python app.py
```
- Khi terminal báo `Application startup complete`, mở trình duyệt web và truy cập vào địa chỉ: **http://localhost:8000**
- Giao diện người dùng sẽ hiện ra, bạn có thể cấp quyền Micro và bấm **"Start Recording"** để hệ thống nhận diện trực tiếp.

### 3. Kiểm thử Hệ thống (Test Pipeline)
Nếu bạn muốn kiểm tra xem toàn bộ các luồng dữ liệu (model nạp có lỗi không, code chạy mượt không) mà không cần mở trình duyệt, hãy chạy:
```bash
python test_pipeline.py
```
Hệ thống sẽ chạy qua file âm thanh mẫu (trong `testing_audio`) với các test case mô phỏng streaming thời gian thực. Nếu cuối cùng in ra `✅ TOÀN BỘ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA THÀNH CÔNG!`, hệ thống của bạn hoàn toàn hoàn hảo.

---

## 📝 Lưu ý Cấu Hình
- **Đổi thiết bị (CPU/GPU)**: Vào file `config.py` và sửa biến `device: str = "cuda"` (để dùng Card đồ họa) hoặc `"cpu"`.
- **Bật/Tắt Module**: Cũng trong `config.py`, bạn có thể bật tắt nhanh các tính năng (VD: `enable_denoise = False`) để làm nhẹ app nếu chạy trên máy tính yếu.
- **Thêm Hotwords**: Chỉ cần mở file `hotwords.txt` và điền thêm các từ ngữ muốn hệ thống nhận diện chính xác (mỗi từ một dòng), sau đó khởi động lại app.
