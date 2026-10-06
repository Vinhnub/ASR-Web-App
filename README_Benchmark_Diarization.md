# Kết Quả Benchmark Phân Đoạn Người Nói (Speaker Diarization)

Báo cáo này trình bày kết quả đo lường và so sánh hiệu suất giữa hai mô hình phân đoạn người nói đang được tích hợp trong hệ thống: **NVIDIA SortFormer (4 speakers)** và **NVIDIA Nemotron-3-Diarization**.

Việc kiểm thử được thực hiện bằng thư viện `pyannote.metrics` dựa trên **50 file âm thanh ngẫu nhiên** (đa người nói, tổng thời lượng ~52.5 phút) từ bộ dữ liệu `ViYT_Diar`. Bộ dữ liệu này có đi kèm nhãn thời gian thực tế chuẩn (Ground Truth RTTM) để đối chiếu chính xác.

---

## Bảng So Sánh Chi Tiết (50 Samples)

| Tiêu chí Đánh giá | NVIDIA SortFormer | NVIDIA Nemotron-3-Diarization | So sánh |
| :--- | :--- | :--- | :--- |
| **Tổng thời lượng Test** | 3149.08 giây (~52.5 phút) | 3149.08 giây (~52.5 phút) | Đồng nhất |
| **Tốc độ (Avg RTF)** | 0.0095 | **0.0037** | **Nemotron-3 nhanh gấp ~2.5 lần** |
| **DER (Diarization Error Rate)**| 20.60% | **12.66%** | **Nemotron-3 giảm 40% lỗi tổng hợp** |
| **Purity (Độ tinh khiết)** | 83.42% | **90.93%** | Nemotron-3 ít nhận diện nhầm người hơn |
| **Coverage (Độ bao phủ)** | **97.64%** | 96.68% | SortFormer nhạy hơn một chút |

---

## Đánh Giá Chuyên Sâu Các Chỉ Số

### 1. Chỉ số DER (Diarization Error Rate)
DER là thước đo quan trọng nhất để đánh giá một mô hình Diarization (bao gồm lỗi do nhận diện sai người, nhận diện sót tiếng, hoặc nhận diện dư tiếng ồn). 
- **Nemotron-3** xuất sắc đạt mức lỗi chỉ **12.66%**, vượt xa so với mức 20.60% của SortFormer. 
- Mức chênh lệch này cho thấy Nemotron-3 xử lý các đoạn hội thoại phức tạp, chuyển lượt nói liên tục, và giọng nói chồng chéo (overlapping speech) tốt hơn rất nhiều.

### 2. Purity (Precision) và Coverage (Recall)
- **Purity (~91% của Nemotron):** Chỉ số này cho thấy khi mô hình đánh dấu một dải âm thanh là của "Người A", thì nó đúng tới 91%. Sortformer dễ bị nhầm lẫn và gộp nhầm giọng người này vào người kia hơn (83.42%).
- **Coverage (~97% của cả 2):** Cả hai mô hình đều có độ bao phủ cực cao. Chúng gần như không bỏ sót bất kỳ một đoạn hội thoại nào của con người trong các file âm thanh thử nghiệm.

### 3. Tốc độ thực thi (RTF - Real Time Factor)
- Nhờ khả năng xử lý vector hoá một lần (không phải giả lập streaming qua các chunk nhỏ), **Nemotron-3** thể hiện sức mạnh vượt trội khi chỉ mất 0.0037 giây để xử lý 1 giây âm thanh. 
- Nói cách khác, để xử lý trọn gói 52.5 phút audio, mô hình chỉ tốn **khoảng 11 giây** chạy Inference trên GPU. Tốc độ này nhanh gấp 2.5 lần so với tiến trình chunking của SortFormer.

---

## Kết Luận & Khuyến Nghị

Dựa trên kết quả đo lường khách quan, **NVIDIA Nemotron-3-Diarization** hoàn toàn áp đảo SortFormer ở cả hai khía cạnh **độ chính xác phân tách** và **tốc độ xử lý Offline**.

💡 **Khuyến nghị sử dụng:**
* Khuyên dùng **Nemotron-3** làm mô hình mặc định cho chế độ Tải file (Upload Video/Audio).
* Trong chế độ Live Microphone theo thời gian thực (Microphone Streaming), SortFormer vẫn là lựa chọn tốt do cấu trúc thiết kế tối ưu riêng cho luồng streaming độ trễ thấp từng chunk nhỏ. Tuy nhiên, nếu server phần cứng mạnh, Nemotron-3 vẫn có thể áp dụng theo chunk lớn.

*(Chi tiết log đo lường từng file có thể tìm thấy tại file `benchmark_results/diarization_metrics_comparison.json`)*
