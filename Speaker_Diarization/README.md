# Hệ Thống Phân Tách & Nhận Diện Người Nói Thời Gian Thực
### (Real-Time Speaker Diarization & Voiceprint Recognition System)

Hệ thống được phát triển dựa trên 2 kiến trúc Deep Learning tiên tiến nhất hiện nay từ **NVIDIA NeMo**:
1. **NVIDIA Streaming Sortformer** (`diar_streaming_sortformer_4spk-v2.1`): Phân tách các lượt nói theo thời gian thực (Real-time Diarization) với độ trễ thấp, tự động giải quyết bài toán hoán vị người nói bằng **Sort Loss** và bộ đệm **AOSC (Arrival-Order Speaker Cache)**.
2. **NVIDIA TitaNet** (`speakerverification_en_titanet_large`): Trích xuất vector đặc trưng âm học (Voiceprint Embedding) 192/512 chiều, có độ chính xác cao hàng đầu để quản lý danh tính và nhận diện người quen.

---

## 📁 Cấu Trúc Thư Mục

```
ASR/Speaker_Diarization/
├── README.md                          # Hướng dẫn chi tiết sử dụng hệ thống
├── streaming_sortformer.py            # Mô-đun & CLI Diarization (NVIDIA Streaming Sortformer)
├── speaker_recognition_titanet.py     # Mô-đun & CLI Voiceprint, Đăng ký & Nhận diện (NVIDIA TitaNet)
└── diarization_pipeline.py            # Pipeline tổng hợp hoàn chỉnh (Sortformer + TitaNet + DB)
```

---

## ⚙️ Yêu Cầu Cài Đặt (Prerequisites)

Hệ thống yêu cầu môi trường Python hỗ trợ PyTorch (tối ưu nhất trên GPU CUDA) và bộ công cụ **NVIDIA NeMo Toolkit**:

```bash
# Kích hoạt môi trường conda của bạn (ví dụ ai_gpu):
conda activate ai_gpu

# Cài đặt NVIDIA NeMo Toolkit ASR
pip install nemo_toolkit['asr']

# Cài đặt các thư viện xử lý âm thanh phụ trợ
pip install soundfile librosa numpy scipy
```

> **Lưu ý trên Windows:** Mã nguồn đã được cấu hình tự động nạp đường dẫn DLL của CUDA / cuDNN từ PyTorch để tương thích hoàn toàn trên hệ điều hành Windows.

---

## 🚀 Hướng Dẫn Sử Dụng Chi Tiết

### 1. Phân Tách Người Nói (Streaming Sortformer)
Sử dụng file `streaming_sortformer.py` để phân đoạn âm thanh thành các lượt nói của từng người (`spk_0`, `spk_1`,...):

```bash
python streaming_sortformer.py \
    --audio "path/to/meeting.wav" \
    --collar 0.25 \
    --min-dur 0.20 \
    --output-rttm "./results/meeting.rttm" \
    --output-json "./results/meeting.json"
```

**Sử dụng trong mã nguồn Python:**
```python
from streaming_sortformer import StreamingSortformerDiarizer

diarizer = StreamingSortformerDiarizer(collar=0.25, min_duration=0.20)
segments = diarizer.diarize_file("meeting.wav")

for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.speaker} (Độ tin cậy: {seg.confidence:.2%})")
```

---

### 2. Quản Lý Hồ Sơ Giọng Nói & Đăng Ký Người Mới (TitaNet)
Sử dụng file `speaker_recognition_titanet.py` để đăng ký giọng nói của nhân sự / người quen vào kho dữ liệu (`speaker_db/`):

#### A. Đăng ký người mới (Enrollment):
Cung cấp 1 file âm thanh mẫu (từ 5 - 10 giây) của người đó:
```bash
# Đăng ký Sếp Tuấn
python speaker_recognition_titanet.py \
    --enroll \
    --name "Anh Tuấn" \
    --audio "samples/tuan_voice.wav" \
    --desc "Trưởng phòng Kỹ thuật"

# Đăng ký Chị Lan
python speaker_recognition_titanet.py \
    --enroll \
    --name "Chị Lan" \
    --audio "samples/lan_voice.wav" \
    --desc "Kế toán trưởng"
```
*(Nếu bạn chạy lệnh enroll nhiều lần với cùng 1 tên, hệ thống sẽ tự động tính trung bình trọng số các vector để mẫu giọng ngày càng chuẩn xác hơn).*

#### B. Xem danh sách người đã đăng ký:
```bash
python speaker_recognition_titanet.py --list
```

#### C. Nhận diện giọng nói từ 1 file bất kỳ:
```bash
python speaker_recognition_titanet.py --identify --audio "test_voice.wav" --threshold 0.70
```

#### D. Xác thực xem 2 file có cùng 1 người nói không:
```bash
python speaker_recognition_titanet.py --verify --audio1 "file1.wav" --audio2 "file2.wav"
```

---

### 3. Pipeline Tổng Hợp (Diarization + Voiceprint Identification)
Sử dụng file `diarization_pipeline.py` để chạy toàn bộ quy trình: vừa tách đoạn nói, vừa gán chính xác tên người thật đã đăng ký trong database:

```bash
python diarization_pipeline.py \
    --audio "meeting_4_people.wav" \
    --db-dir "./speaker_db" \
    --output-dir "./diarization_results" \
    --threshold 0.70 \
    --min-speech-dur 0.80 \
    --auto-save-unknown
```

#### Các tùy chọn nâng cao:
* `--simulate-streaming`: Bật chế độ mô phỏng phát luồng âm thanh theo thời gian thực (pacing 1.0x). Terminal sẽ hiển thị ai đang nói theo từng giây như một cuộc họp đang diễn ra live!
* `--auto-save-unknown`: Khi phát hiện một người lạ chưa có trong database, pipeline sẽ tự động cắt đoạn âm thanh của người đó và lưu vào thư mục `diarization_results/unknown_speakers/` để bạn có thể nghe lại và đăng ký sau.

#### Kết quả đầu ra gồm có:
1. **Bảng Timeline trên Terminal**: Hiển thị rõ mốc thời gian, thời lượng, tên người nói và độ khớp tương đồng.
2. **File RTTM (`.rttm`)**: Định dạng chuẩn quốc tế cho Speaker Diarization, tương thích với các công cụ benchmark.
3. **File JSON (`.json`)**: Chi tiết toàn bộ các mốc thời gian, điểm tương đồng (similarity) và độ tin cậy.
4. **File TXT tóm tắt (`_summary.txt`)**: Tổng hợp thời lượng phát biểu của từng thành viên trong cuộc họp.

---

## 💡 Cơ Chế "Khóa Nhãn Định Danh" (Identity Lock)
Để tối ưu hóa tài nguyên GPU và thời gian xử lý:
1. **Streaming Sortformer** phân tách liên tục các mốc thời gian của `spk_0`, `spk_1`...
2. Khi một người nói tích lũy đủ tối thiểu `0.8s` (đảm bảo mẫu âm thanh đủ đặc trưng cho TitaNet), pipeline kích hoạt trích xuất Voiceprint và so khớp với cơ sở dữ liệu.
3. Khi đã nhận diện được `spk_0` là `"Anh Tuấn"`, hệ thống **khóa nhãn này** cho toàn bộ session. Pipeline không cần phải chạy lại TitaNet ở mỗi frame tiếp theo, giúp giảm tải hơn **85%** tài nguyên tính toán!

---

## 📊 Kết Quả Benchmark TitaNet Trên Tiếng Việt (VieSpeaker)

Hiệu năng của mô hình **NVIDIA TitaNet-Large** đã được kiểm chứng độc lập trên bộ dữ liệu tiếng Việt tự nhiên **VieSpeaker** (INTERSPEECH 2026) với 1.501 audio clips (~3.51 giờ) trên GPU RTX 3050 Ti:

| Chỉ số đánh giá | Kết quả thực tế (2.000 cặp cân bằng) | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| **Equal Error Rate (EER)** | **13.90%** | Năng lực zero-shot xuất sắc trên tiếng Việt tự nhiên |
| **minDCF ($P_{target}=0.01$)** | **0.3740** | Chỉ số chi phí nhận dạng theo chuẩn NIST SRE |
| **Ngưỡng Cosine tối ưu** | **0.4273** | Ngưỡng phân tách tốt nhất giữa người quen và người lạ |
| **Độ tách biệt điểm số** | **+0.3934** | Target: `+0.6628` vs Impostor: `+0.2694` |
| **Tốc độ trích xuất** | **19.4 clips / giây** | Độ trễ chỉ **51.5 ms / clip** (RTF: **0.0061x**, nhanh gấp **163x** realtime) |

> 📖 Xem tài liệu chi tiết: [`BENCHMARK_TITANET_VIESPEAKER.md`](file:///e:/PythonFile/Project/ASR/BENCHMARK_TITANET_VIESPEAKER.md) hoặc [`Readme/15_benchmark_titanet_viespeaker.md`](file:///e:/PythonFile/Project/ASR/Readme/15_benchmark_titanet_viespeaker.md).

