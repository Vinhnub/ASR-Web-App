# -*- coding: utf-8 -*-
"""
Module Tóm Tắt Nội Dung Tự Động (AI Summarization) sử dụng Qwen2.5-1.5B-Instruct.
Hỗ trợ tóm tắt biên bản cuộc họp, hội thoại và tài liệu ASR theo từng người nói và các ý chính.
Hoạt động linh hoạt trên cả GPU (VRAM < 2GB) và CPU.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class SummaryModule:
    """Module tóm tắt nội dung sử dụng Qwen2.5-1.5B-Instruct."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        enabled: bool = True,
        max_tokens: int = 384,
        temperature: float = 0.3,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Tự động phát hiện cấu hình phần cứng: nếu GPU VRAM < 6GB (như RTX 3050 4GB),
        # chạy Qwen2.5 trên CPU để dành trọn VRAM cho NeMo Diarization và TitaNet, triệt tiêu lỗi tràn VRAM (Thrashing).
        if device == "cuda" and torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_vram_gb < 6.0:
                    print(f"[SummaryModule] GPU VRAM ({total_vram_gb:.1f}GB < 6.0GB) -> Tự động chuyển Qwen2.5 sang CPU đa luồng để tránh nghẽn VRAM!")
                    self.device = "cpu"
                else:
                    self.device = "cuda"
            except Exception:
                self.device = "cuda"
        else:
            self.device = "cpu"


        self.model = None
        self.tokenizer = None
        self._is_loading = False

    def _ensure_model_loaded(self) -> bool:
        """Nạp mô hình Qwen2.5 theo cơ chế Lazy Loading khi có yêu cầu đầu tiên."""
        if not self.enabled:
            return False
        if self.model is not None and self.tokenizer is not None:
            return True
        if self._is_loading:
            return False

        self._is_loading = True
        try:
            print(f"[SummaryModule] Đang khởi tạo mô hình '{self.model_name}' trên {self.device.upper()}...")
            t0 = time.time()
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            elapsed = time.time() - t0
            print(f"[SummaryModule] Nạp mô hình Qwen2.5 thành công trong {elapsed:.2f}s!")
            return True
        except Exception as e:
            print(f"[SummaryModule] Lỗi khi nạp mô hình Qwen2.5: {e}")
            self.model = None
            self.tokenizer = None
            return False
        finally:
            self._is_loading = False

    def summarize(
        self,
        text: str = "",
        segments: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Tạo bản tóm tắt nội dung súc tích từ văn bản hoặc danh sách phân đoạn người nói.

        Tham số:
            text: Văn bản toàn bài (final_text)
            segments: Danh sách phân đoạn [{'speaker': ..., 'text': ...}]
            max_tokens: Số token tối đa cần sinh
            temperature: Độ ngẫu nhiên (mặc định 0.3 để tóm tắt chuẩn xác)
        """
        if not self.enabled:
            return ""

        # Chuẩn bị nội dung đầu vào
        lines = []
        if segments and len(segments) > 0:
            for seg in segments:
                spk = seg.get("speaker") or "Người nói"
                seg_text = (seg.get("text") or "").strip()
                if seg_text:
                    lines.append(f"- {spk}: {seg_text}")

        if lines:
            formatted_input = "\n".join(lines)
        else:
            formatted_input = (text or "").strip()

        word_count = len(formatted_input.split())
        if word_count < 10:
            return "Văn bản quá ngắn để tạo bản tóm tắt ý chính."

        # Đảm bảo mô hình đã được nạp
        if not self._ensure_model_loaded():
            return "Không thể nạp mô hình tóm tắt Qwen2.5 (Vui lòng kiểm tra kết nối mạng hoặc bộ nhớ GPU)."

        # Cấu trúc Prompt chuyên dụng tạo Biên Bản Cuộc Họp (Meeting Minutes - MM)
        system_prompt = (
            "Bạn là một thư ký AI chuyên nghiệp ghi biên bản cuộc họp (Meeting Minutes). "
            "Nhiệm vụ của bạn là đọc kỹ nội dung hội thoại cuộc họp dưới đây, làm sạch nội dung "
            "và viết lại thành một bản biên bản cuộc họp hoàn chỉnh bằng tiếng Việt.\n\n"
            "Bắt buộc tuân thủ chính xác cấu trúc 6 phần sau đây (sử dụng đúng các biểu tượng emoji và tiêu đề):\n\n"
            "### 🎯 Mục tiêu cuộc họp\n"
            "(Nêu rõ mục đích, lý do tổ chức hoặc nội dung cốt lõi của cuộc họp trong 1-2 câu)\n\n"
            "### 📑 Nội dung chính\n"
            "(Tóm tắt các vấn đề trọng tâm và nội dung thảo luận quan trọng dưới dạng các gạch đầu dòng '- ')\n\n"
            "### ✅ Kết luận\n"
            "(Các kết luận, quyết định hoặc thỏa thuận thống nhất cuối cùng)\n\n"
            "### 📋 Action items\n"
            "(Danh sách các hành động/đầu việc cụ thể cần thực hiện tiếp theo dưới dạng gạch đầu dòng '- ')\n\n"
            "### 👤 Người phụ trách\n"
            "(Ghi rõ ai phụ trách công việc gì hoặc vai trò của từng người; nếu không có thông tin cụ thể ghi 'Chưa chỉ định')\n\n"
            "### 📅 Deadline\n"
            "(Thời hạn/ngày tháng hoàn thành các công việc nếu có đề cập; nếu không đề cập ghi 'Chưa xác định')\n\n"
            "Lưu ý: Viết súc tích, chuyên nghiệp, khách quan, trung thực với nội dung gốc, không bịa đặt thông tin."
        )

        user_prompt = (
            f"Dưới đây là nội dung cuộc họp cần ghi biên bản:\n\n{formatted_input}\n\n"
            "Hãy viết lại thành Biên bản cuộc họp (Meeting Minutes) theo đúng cấu trúc 6 phần trên:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            prompt_formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([prompt_formatted], return_tensors="pt").to(self.device)

            gen_tokens = max_tokens or self.max_tokens
            temp = temperature if temperature is not None else self.temperature

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_tokens,
                    temperature=temp,
                    do_sample=(temp > 0.0),
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

            # Trích xuất phần văn bản sinh ra
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            summary_result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return summary_result

        except Exception as e:
            print(f"[SummaryModule] Lỗi khi sinh tóm tắt: {e}")
            return f"Lỗi trong quá trình tạo tóm tắt: {str(e)}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_parameters(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """Cập nhật linh hoạt tham số sinh từ."""
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        if temperature is not None:
            self.temperature = float(temperature)

    def set_enabled(self, enabled: bool):
        """Bật / Tắt module tóm tắt."""
        self.enabled = enabled
