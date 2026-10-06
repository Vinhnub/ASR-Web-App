# Báo Cáo Benchmark ASR: Zipformer Streaming (30M) vs Zipformer Offline (150M)

Tài liệu này tổng hợp kết quả đánh giá hiệu năng (benchmark) so sánh giữa hai mô hình nhận dạng giọng nói tự động (ASR) tiếng Việt:
1. **Bản Streaming (30M tham số):** `hynt/Zipformer-30M-RNNT-Streaming-6000h` (Mô hình trực tuyến, xử lý chunk-by-chunk với cơ chế Cache).
2. **Bản Offline (150M tham số):** `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h` (Mô hình ngoại tuyến, đã được thay thế theo yêu cầu, xử lý theo Sliding-Window).

Benchmark được thực hiện tự động trên tập dữ liệu **ViMedCSS_Test** với **30 mẫu âm thanh** ngẫu nhiên, chạy trên môi trường CPU/GPU tiêu chuẩn.

## 1. Bảng So Sánh Hiệu Năng (Comparison Table)

| Tiêu chí Đánh giá | Zipformer Streaming (30M) | Zipformer Offline (150M) |
| :--- | :--- | :--- |
| **Số lượng Samples test** | 30 file | 30 file |
| **Tốc độ xử lý (Avg RTF)** | `0.1800` | `0.0272` **(Siêu Nhanh)** |
| **Độ lỗi ký tự (Avg CER)** | 25.64% | **19.24% (Chính xác hơn)** |
| **Cơ chế hoạt động** | True Streaming ASR + Cache | Offline Sliding-Window |

*(Ghi chú: Giá trị RTF (Real-Time Factor) càng thấp chứng tỏ mô hình suy luận càng nhanh. CER (Character Error Rate) càng thấp chứng tỏ mô hình nhận dạng càng chính xác.)*

## 2. Phân Tích & Nhận Xét Chuyên Sâu

### 2.1. Độ Chính Xác (Accuracy / CER)
* **Zipformer Offline (150M)** cho kết quả chính xác hơn rõ rệt với tỉ lệ lỗi ký tự (CER) giảm xuống chỉ còn 19.24% so với 25.64% của bản Streaming. Điều này là hoàn toàn dễ hiểu vì mô hình Offline sở hữu số lượng tham số lớn gấp 5 lần (150M vs 30M), giúp nó học được nhiều đặc trưng ngôn ngữ phong phú hơn. Hơn nữa, với cơ chế Offline, mô hình có thể nhìn được trọn vẹn ngữ cảnh của câu (Full Context) thay vì chỉ nhìn ngữ cảnh quá khứ như bản Streaming.

### 2.2. Tốc Độ Xử Lý (Speed / RTF)
* Mô hình Offline có tốc độ tính toán (RTF: `0.0272`) nhanh hơn so với bản Streaming (`0.1800`). Nguyên nhân là do ở bản Offline, toàn bộ chunk audio (sliding window 2.0s) được truyền vào mạng Neural để tính toán đồng loạt thành một mẻ (batch processing). Tuy nhiên, cần hiểu rằng bản Offline yêu cầu phải nhận đủ một đoạn âm thanh dài rồi mới bắt đầu dịch (độ trễ ban đầu lớn). 
* Trong khi đó, bản **Streaming** dịch theo từng gói nhỏ (mỗi 0.32s) thời gian thực, có độ trễ cục bộ (latency per chunk) cực nhỏ và phù hợp để kết nối trực tiếp với microphone hoặc WebSocket, dù rằng tổng thời gian tính qua các vòng lặp bị kéo dài ra.

### 2.3. Lời Khuyên Áp Dụng (Use Cases)
* **Dùng bản Offline 150M:** Phù hợp với các hệ thống cần giải mã file ghi âm (Bóc băng Audio/Video), phân tích hồ sơ bệnh án hoặc các tác vụ không quá khắt khe về độ trễ, yêu cầu độ chính xác cao nhất có thể.
* **Dùng bản Streaming 30M:** Phù hợp bắt buộc với các ứng dụng trợ lý ảo giao tiếp theo thời gian thực (Live Stream, Live Dictation), robot hoặc các kiosk điều khiển bằng giọng nói nơi người dùng nói đến đâu chữ hiện ra đến đó.

---
**Kết luận:** Quá trình thay thế model offline mặc định bằng mô hình `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h` đã hoàn tất thành công trong mã nguồn dự án. Kết quả cho thấy mô hình mới đáp ứng rất tốt trên tập dữ liệu kiểm thử thực tế.

*Dữ liệu raw chi tiết JSON có thể xem tại: `benchmark_results/models_comparison_zipformer.json`*
