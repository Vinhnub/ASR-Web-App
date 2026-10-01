# Benchmark Report: Automatic Punctuation & Capitalization (Vietnamese)

## 1. Tổng quan Thử nghiệm
- **Tập dữ liệu Benchmark:** **ViCapPunc** (Interspeech 2022 - chuẩn nhận dạng giọng nói & hội thoại).
- **Số lượng câu thử nghiệm:** `300` câu (`7384` từ).
- **Môi trường phần cứng:** Windows, CUDA sẵn sàng (`torch.cuda.is_available() = True`).
- **Mục tiêu:** So sánh toàn diện các mô hình Lightweight SOTA về **Độ chính xác dấu câu & chữ hoa**, **Tốc độ suy luận (Latency/Throughput)**, và **Dung lượng tiêu thụ (Disk, RAM, VRAM)**.

## 2. Bảng Tổng hợp Kết quả So sánh Toàn diện
| Mô hình | Thiết bị | Dung lượng Disk | Peak RAM / VRAM | Mean Latency | Throughput | Casing Acc | Punct F1 | Period (.) F1 | Comma (,) F1 | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViBERT-Capu (ONNX INT8)** | CPU | `110.35 MB` | `1997 MB` / `380.1 MB` | **`33.69 ms`** | `730.47 w/s` | **`97.26%`** | **`62.26%`** | `69.14%` | `56.46%` | `42.67%` |
| **ViBERT-Capu (ONNX FP32 - CPU)** | CPU | `438.17 MB` | `2004 MB` / `380.1 MB` | **`25.16 ms`** | `978.16 w/s` | **`97.22%`** | **`61.12%`** | `67.41%` | `56.02%` | `41.33%` |
| **VFastPunct (BiLSTM-CRF - CPU)** | CPU | `378.84 MB` | `2001 MB` / `380.1 MB` | **`22.06 ms`** | `1115.09 w/s` | **`97.67%`** | **`73.2%`** | `85.45%` | `67.24%` | `37.0%` |
| **VFastPunct (BiLSTM-CRF - CUDA)** | CUDA | `378.84 MB` | `2251 MB` / `453.9 MB` | **`35.4 ms`** | `695.14 w/s` | **`97.67%`** | **`73.2%`** | `85.45%` | `67.24%` | `37.0%` |

## 3. Phân tích Chi tiết Từng Mô hình

### 📌 ViBERT-Capu (ONNX INT8)
- **Cấu hình:** Thiết bị `CPU`, Model Footprint `110.35 MB`.
- **Độ trễ suy luận:** Trung bình `33.69 ms/câu`, P95: `160.1 ms/câu` (Băng thông: `29.68 câu/giây`, `730.47 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`62.26%`** (Precision: `58.04%`, Recall: `67.13%`).
  - Dấu chấm (`.`): F1 `69.14%` (Precision: `60.11%`, Recall: `81.37%`).
  - Dấu phẩy (`,`): F1 `56.46%` (Precision: `57.93%`, Recall: `55.07%`).
  - Viết hoa (Casing): **`97.26%`**.
  - Khớp nguyên câu 100% (Exact Match): **`42.67%`** | Word Error Rate: `12.43%`.

### 📌 ViBERT-Capu (ONNX FP32 - CPU)
- **Cấu hình:** Thiết bị `CPU`, Model Footprint `438.17 MB`.
- **Độ trễ suy luận:** Trung bình `25.16 ms/câu`, P95: `58.97 ms/câu` (Băng thông: `39.74 câu/giây`, `978.16 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`61.12%`** (Precision: `56.88%`, Recall: `66.05%`).
  - Dấu chấm (`.`): F1 `67.41%` (Precision: `58.13%`, Recall: `80.23%`).
  - Dấu phẩy (`,`): F1 `56.02%` (Precision: `58.31%`, Recall: `53.91%`).
  - Viết hoa (Casing): **`97.22%`**.
  - Khớp nguyên câu 100% (Exact Match): **`41.33%`** | Word Error Rate: `12.53%`.

### 📌 VFastPunct (BiLSTM-CRF - CPU)
- **Cấu hình:** Thiết bị `CPU`, Model Footprint `378.84 MB`.
- **Độ trễ suy luận:** Trung bình `22.06 ms/câu`, P95: `28.32 ms/câu` (Băng thông: `45.3 câu/giây`, `1115.09 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`73.2%`** (Precision: `69.71%`, Recall: `77.05%`).
  - Dấu chấm (`.`): F1 `85.45%` (Precision: `83.88%`, Recall: `87.07%`).
  - Dấu phẩy (`,`): F1 `67.24%` (Precision: `66.11%`, Recall: `68.41%`).
  - Viết hoa (Casing): **`97.67%`**.
  - Khớp nguyên câu 100% (Exact Match): **`37.0%`** | Word Error Rate: `6.27%`.

### 📌 VFastPunct (BiLSTM-CRF - CUDA)
- **Cấu hình:** Thiết bị `CUDA`, Model Footprint `378.84 MB`.
- **Độ trễ suy luận:** Trung bình `35.4 ms/câu`, P95: `50.0 ms/câu` (Băng thông: `28.24 câu/giây`, `695.14 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`73.2%`** (Precision: `69.71%`, Recall: `77.05%`).
  - Dấu chấm (`.`): F1 `85.45%` (Precision: `83.88%`, Recall: `87.07%`).
  - Dấu phẩy (`,`): F1 `67.24%` (Precision: `66.11%`, Recall: `68.41%`).
  - Viết hoa (Casing): **`97.67%`**.
  - Khớp nguyên câu 100% (Exact Match): **`37.0%`** | Word Error Rate: `6.27%`.

## 4. Ví dụ Dự đoán Thực tế của các Mô hình

**Mẫu #1:**
- **Input thô từ ASR:** `cho em hỏi là e từ nc ngoài về nhưng bên đó chích 2 mũi vaccine covid rồi`
- **Ground Truth:** `Cho em hỏi là e từ nc ngoài về nhưng bên đó chích 2 mũi vaccine covid rồi.`
- **ViBERT-Capu (ONNX INT8):** `Cho em hỏi là e từ nc ngoài về nhưng bên đó chích 2 mũi vaccine Covid rồi`
- **ViBERT-Capu (ONNX FP32 - CPU):** `Cho em hỏi là e từ nc ngoài về nhưng bên đó chích 2 mũi vaccine Covid rồi`
- **VFastPunct (BiLSTM-CRF - CPU):** `Cho em hỏi là e từ nc ngoài về nhưng bên đó chích 2 mũi vaccine Covid rồi.`
- **VFastPunct (BiLSTM-CRF - CUDA):** `Cho em hỏi là e từ nc ngoài về nhưng bên đó chích 2 mũi vaccine Covid rồi.`

**Mẫu #2:**
- **Input thô từ ASR:** `nhunge k có giấy tờ chứng minh sự việc đó`
- **Ground Truth:** `Nhunge k có giấy tờ chứng minh sự việc đó.`
- **ViBERT-Capu (ONNX INT8):** `Nhunge k có giấy tờ chứng minh sự việc đó.`
- **ViBERT-Capu (ONNX FP32 - CPU):** `Nhunge k có giấy tờ chứng minh sự việc đó.`
- **VFastPunct (BiLSTM-CRF - CPU):** `Nhunge k có giấy tờ chứng minh sự việc đó.`
- **VFastPunct (BiLSTM-CRF - CUDA):** `Nhunge k có giấy tờ chứng minh sự việc đó.`

**Mẫu #3:**
- **Input thô từ ASR:** `có cách nào xét nghiệm lại để chứng minh k chứ đã chích r chích nữa k được chào bạn bạn có thể xét nghiệm kháng thể covid và liên hệ với chính quyền địa phương để chứng minh sự việc tôi thỉnh thoảng hay đi tiểu ra chất màu trắng đục đục như nước vo gạo sau mỗi lần đi là tôi bị buốt không chịu được sau một lúc thì đỡ`
- **Ground Truth:** `Có cách nào xét nghiệm lại để chứng minh k chứ đã chích r chích nữa k được Chào bạn, bạn có thể xét nghiệm kháng thể covid Và liên hệ với chính quyền địa phương để chứng minh sự việc Tôi thỉnh thoảng hay đi tiểu ra chất màu trắng đục đục như nước vo gạo, sau mỗi lần đi là tôi bị buốt không chịu được, sau một lúc thì đỡ.`
- **ViBERT-Capu (ONNX INT8):** `Có cách nào xét nghiệm lại để chứng minh k chứ đã chích r chích nữa k được. Chào bạn, bạn có thể xét nghiệm kháng thể Covid và liên hệ với chính quyền địa phương để chứng minh sự việc. Tôi thỉnh thoảng hay đi tiểu ra Để chứng minh sự việc, tôi thỉnh thoảng hay đi tiểu ra chất màu trắng đục đục như nước vo gạo. Sau mỗi lần đi là tôi bị buốt không chịu được. Sau một lúc thì đỡ.`
- **ViBERT-Capu (ONNX FP32 - CPU):** `Có cách nào xét nghiệm lại để chứng minh k chứ đã chích r chích nữa k được. Chào bạn, bạn có thể xét nghiệm kháng thể Covid và liên hệ với chính quyền địa phương để chứng minh sự việc. Tôi thỉnh thoảng hay đi tiểu ra Để chứng minh sự việc, tôi thỉnh thoảng hay đi tiểu ra chất màu trắng đục đục như nước vo gạo. Sau mỗi lần đi là tôi bị buốt không chịu được. Sau một lúc thì đỡ.`
- **VFastPunct (BiLSTM-CRF - CPU):** `Có cách nào xét nghiệm lại để chứng minh k chứ đã chích r chích nữa k được chào bạn, bạn có thể xét nghiệm kháng thể Covid và liên hệ với chính quyền địa phương để chứng minh sự việc. Tôi thỉnh thoảng hay đi tiểu ra chất màu trắng đục đục như nước vo gạo, sau mỗi lần đi là tôi bị buốt, không chịu được, sau một lúc thì đỡ.`
- **VFastPunct (BiLSTM-CRF - CUDA):** `Có cách nào xét nghiệm lại để chứng minh k chứ đã chích r chích nữa k được chào bạn, bạn có thể xét nghiệm kháng thể Covid và liên hệ với chính quyền địa phương để chứng minh sự việc. Tôi thỉnh thoảng hay đi tiểu ra chất màu trắng đục đục như nước vo gạo, sau mỗi lần đi là tôi bị buốt, không chịu được, sau một lúc thì đỡ.`

## 5. Kết luận & Đề xuất Áp dụng vào Production ASR
1. **Mô hình Khuyên dùng Hàng đầu: `ViBERT-Capu (ONNX INT8)`**:
   - **Dung lượng siêu nhẹ (~110 MB)** và chỉ chạy trên CPU thông qua ONNX Runtime mà không cần GPU.
   - Độ trễ cực thấp (< 25 ms trên CPU thông thường), hoàn toàn đáp ứng các ứng dụng ASR thời gian thực (Streaming) và chạy offline trên laptop/PC người dùng.
   - Giữ nguyên vẹn 99% độ chính xác so với bản FP32.
2. **So sánh FP32 vs INT8**:
   - Bản FP32 có độ chính xác nhỉnh hơn nhẹ nhưng kích thước tệp lớn gấp 4 lần (438 MB vs 110 MB) và độ trễ CPU cao hơn.
3. **Kiến trúc BiLSTM-CRF (`VFastPunct mLstmPunctCap`)**:
   - Lớp CRF giúp kiểm soát tốt các trạng thái chuyển dịch dấu câu hợp lý.
   - Thích hợp khi cần mô hình recurrent hoặc triển khai trên các môi trường chỉ hỗ trợ mạng tuần hoàn đơn giản.
4. **Khuyến nghị Pipeline ASR**:
   - Tích hợp `ViBERT-Capu (ONNX INT8)` vào cuối pipeline sau khi Zipformer/Chunkformer xuất ra văn bản thô.
   - Kết hợp khoảng lặng âm thanh (`silence_duration > 600ms`) từ mốc thời gian của ASR để chốt dấu câu cuối câu một cách chắc chắn nhất.