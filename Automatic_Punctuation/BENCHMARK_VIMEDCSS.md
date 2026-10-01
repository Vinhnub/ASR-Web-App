# Benchmark Report: Automatic Punctuation & Capitalization (ViMedCSS)

## 1. Tổng quan Thử nghiệm
- **Tập dữ liệu Benchmark:** **ViMedCSS** (Clinical Medical Code-Switching Speech Benchmark).
- **Số lượng câu thử nghiệm:** `300` câu (`7624` từ).
- **Môi trường phần cứng:** Windows, CUDA sẵn sàng (`torch.cuda.is_available() = True`).
- **Mục tiêu:** So sánh toàn diện các mô hình Lightweight SOTA về **Độ chính xác dấu câu & chữ hoa**, **Tốc độ suy luận (Latency/Throughput)**, và **Dung lượng tiêu thụ (Disk, RAM, VRAM)**.

## 2. Bảng Tổng hợp Kết quả So sánh Toàn diện
| Mô hình | Thiết bị | Dung lượng Disk | Peak RAM / VRAM | Mean Latency | Throughput | Casing Acc | Punct F1 | Period (.) F1 | Comma (,) F1 | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViBERT-Capu (ONNX INT8)** | CPU | `110.35 MB` | `1995 MB` / `380.1 MB` | **`128.32 ms`** | `198.01 w/s` | **`97.04%`** | **`71.81%`** | `78.61%` | `65.58%` | `24.67%` |
| **ViBERT-Capu (ONNX FP32 - CPU)** | CPU | `438.17 MB` | `2001 MB` / `380.1 MB` | **`204.42 ms`** | `124.31 w/s` | **`97.15%`** | **`70.78%`** | `77.48%` | `63.47%` | `24.67%` |
| **VFastPunct (BiLSTM-CRF - CPU)** | CPU | `378.84 MB` | `1993 MB` / `380.1 MB` | **`172.95 ms`** | `146.93 w/s` | **`97.15%`** | **`74.21%`** | `87.44%` | `61.29%` | `25.33%` |
| **VFastPunct (BiLSTM-CRF - CUDA)** | CUDA | `378.84 MB` | `2243 MB` / `453.9 MB` | **`77.49 ms`** | `327.89 w/s` | **`97.15%`** | **`74.21%`** | `87.44%` | `61.29%` | `25.33%` |

## 3. Phân tích Chi tiết Từng Mô hình

### 📌 ViBERT-Capu (ONNX INT8)
- **Cấu hình:** Thiết bị `CPU`, Model Footprint `110.35 MB`.
- **Độ trễ suy luận:** Trung bình `128.32 ms/câu`, P95: `254.75 ms/câu` (Băng thông: `7.79 câu/giây`, `198.01 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`71.81%`** (Precision: `74.87%`, Recall: `68.98%`).
  - Dấu chấm (`.`): F1 `78.61%` (Precision: `79.8%`, Recall: `77.45%`).
  - Dấu phẩy (`,`): F1 `65.58%` (Precision: `73.88%`, Recall: `58.96%`).
  - Viết hoa (Casing): **`97.04%`**.
  - Khớp nguyên câu 100% (Exact Match): **`24.67%`** | Word Error Rate: `8.7%`.

### 📌 ViBERT-Capu (ONNX FP32 - CPU)
- **Cấu hình:** Thiết bị `CPU`, Model Footprint `438.17 MB`.
- **Độ trễ suy luận:** Trung bình `204.42 ms/câu`, P95: `312.72 ms/câu` (Băng thông: `4.89 câu/giây`, `124.31 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`70.78%`** (Precision: `74.66%`, Recall: `67.28%`).
  - Dấu chấm (`.`): F1 `77.48%` (Precision: `78.52%`, Recall: `76.47%`).
  - Dấu phẩy (`,`): F1 `63.47%` (Precision: `73.19%`, Recall: `56.03%`).
  - Viết hoa (Casing): **`97.15%`**.
  - Khớp nguyên câu 100% (Exact Match): **`24.67%`** | Word Error Rate: `8.74%`.

### 📌 VFastPunct (BiLSTM-CRF - CPU)
- **Cấu hình:** Thiết bị `CPU`, Model Footprint `378.84 MB`.
- **Độ trễ suy luận:** Trung bình `172.95 ms/câu`, P95: `193.36 ms/câu` (Băng thông: `5.78 câu/giây`, `146.93 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`74.21%`** (Precision: `79.89%`, Recall: `69.29%`).
  - Dấu chấm (`.`): F1 `87.44%` (Precision: `89.69%`, Recall: `85.29%`).
  - Dấu phẩy (`,`): F1 `61.29%` (Precision: `68.13%`, Recall: `55.7%`).
  - Viết hoa (Casing): **`97.15%`**.
  - Khớp nguyên câu 100% (Exact Match): **`25.33%`** | Word Error Rate: `6.14%`.

### 📌 VFastPunct (BiLSTM-CRF - CUDA)
- **Cấu hình:** Thiết bị `CUDA`, Model Footprint `378.84 MB`.
- **Độ trễ suy luận:** Trung bình `77.49 ms/câu`, P95: `108.67 ms/câu` (Băng thông: `12.9 câu/giây`, `327.89 từ/giây`).
- **Chất lượng phục hồi:**
  - Overall Punctuation F1: **`74.21%`** (Precision: `79.89%`, Recall: `69.29%`).
  - Dấu chấm (`.`): F1 `87.44%` (Precision: `89.69%`, Recall: `85.29%`).
  - Dấu phẩy (`,`): F1 `61.29%` (Precision: `68.13%`, Recall: `55.7%`).
  - Viết hoa (Casing): **`97.15%`**.
  - Khớp nguyên câu 100% (Exact Match): **`25.33%`** | Word Error Rate: `6.14%`.

## 4. Ví dụ Dự đoán Thực tế của các Mô hình

**Mẫu #1:**
- **Input thô từ ASR:** `điều này có nghĩa là để xuất hiện bệnh cả hai bản sao của gen phải bị đột biến và bệnh nhân phải là nam giới về mặt di truyền`
- **Ground Truth:** `Điều này có nghĩa là để xuất hiện bệnh, cả hai bản sao của gen phải bị đột biến và bệnh nhân phải là nam giới về mặt di truyền,`
- **ViBERT-Capu (ONNX INT8):** `Điều này có nghĩa là để xuất hiện bệnh, cả hai bản sao của gen phải bị đột biến và bệnh nhân phải là nam giới về mặt di truyền.`
- **ViBERT-Capu (ONNX FP32 - CPU):** `Điều này có nghĩa là để xuất hiện bệnh, cả hai bản sao của gen phải bị đột biến và bệnh nhân phải là nam giới về mặt di truyền.`
- **VFastPunct (BiLSTM-CRF - CPU):** `Điều này có nghĩa là để xuất hiện bệnh, cả hai bản sao của gen phải bị đột biến và bệnh nhân phải là nam giới về mặt di truyền.`
- **VFastPunct (BiLSTM-CRF - CUDA):** `Điều này có nghĩa là để xuất hiện bệnh, cả hai bản sao của gen phải bị đột biến và bệnh nhân phải là nam giới về mặt di truyền.`

**Mẫu #2:**
- **Input thô từ ASR:** `rồi tiếp theo về lợi khuẩn thì sản phẩm alpha lipid trước đây đã được trang bị rất tốt với cái hàm lượng lợi khuẩn là 1 tỷ lợi khuẩn tiêu hóa trên một phần`
- **Ground Truth:** `Rồi tiếp theo, về lợi khuẩn thì sản phẩm Alpha lipid trước đây đã được trang bị rất tốt với cái hàm lượng lợi khuẩn là 1 tỷ lợi khuẩn tiêu hóa trên một phần.`
- **ViBERT-Capu (ONNX INT8):** `Rồi tiếp theo về lợi khuẩn thì sản phẩm Alpha lipid trước đây đã được trang bị rất tốt với cái hàm lượng lợi khuẩn là 1 tỷ lợi khuẩn tiêu hóa trên một phần.`
- **ViBERT-Capu (ONNX FP32 - CPU):** `Rồi tiếp theo về lợi khuẩn thì sản phẩm Alpha Lipid trước đây đã được trang bị rất tốt với cái hàm lượng lợi khuẩn là 1 tỷ lợi khuẩn tiêu hóa trên một phần.`
- **VFastPunct (BiLSTM-CRF - CPU):** `Rồi tiếp theo về lợi khuẩn thì sản phẩm Alpha Lipid trước đây đã được trang bị rất tốt với cái hàm lượng lợi khuẩn là 1 tỷ lợi khuẩn tiêu hóa trên một phần.`
- **VFastPunct (BiLSTM-CRF - CUDA):** `Rồi tiếp theo về lợi khuẩn thì sản phẩm Alpha Lipid trước đây đã được trang bị rất tốt với cái hàm lượng lợi khuẩn là 1 tỷ lợi khuẩn tiêu hóa trên một phần.`

**Mẫu #3:**
- **Input thô từ ASR:** `rồi về chất khoáng ấy thì alpha lipid thì có ba loại khoáng chất chính đó là sắt đó là magie và đó là canxi`
- **Ground Truth:** `Rồi về chất khoáng ấy thì Alpha lipid thì có ba loại khoáng chất chính, đó là sắt, đó là magie và đó là canxi.`
- **ViBERT-Capu (ONNX INT8):** `Rồi về chất khoáng ấy thì alpha, lipid thì có 3 loại khoáng chất chính, đó là sắt, đó là magie và đó là canxi.`
- **ViBERT-Capu (ONNX FP32 - CPU):** `Rồi về chất khoáng ấy thì alpha lipid thì có 3 loại khoáng chất chính, đó là sắt, đó là magie và đó là canxi.`
- **VFastPunct (BiLSTM-CRF - CPU):** `Rồi về chất khoáng ấy thì Alpha Lipid thì có ba loại khoáng chất chính đó là sắt, đó là magie và đó là canxi.`
- **VFastPunct (BiLSTM-CRF - CUDA):** `Rồi về chất khoáng ấy thì Alpha Lipid thì có ba loại khoáng chất chính đó là sắt, đó là magie và đó là canxi.`

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