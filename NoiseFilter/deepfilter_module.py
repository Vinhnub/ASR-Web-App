import os
import sys
import time
import zipfile
from typing import Optional, Union
import numpy as np
import torch

# Thiết lập UTF-8 cho Windows console để tránh UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo đường dẫn tới DeepFilterNet được nhận diện
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DF_PKG_DIR = os.path.join(CURRENT_DIR, "DeepFilterNet", "DeepFilterNet")
if DF_PKG_DIR not in sys.path:
    sys.path.insert(0, DF_PKG_DIR)

try:
    from df.enhance import init_df, enhance, resample
    from df.io import load_audio as df_load_audio, save_audio as df_save_audio
except ImportError as e:
    raise ImportError(
        f"Không thể import DeepFilterNet từ '{DF_PKG_DIR}'. "
        f"Vui lòng đảm bảo đã cài đặt deepfilterlib và loguru. Lỗi chi tiết: {e}"
    )


class DeepFilter:
    """
    Module Noise Filtering sử dụng DeepFilterNet (mặc định DeepFilterNet3).
    Được tinh chỉnh để tích hợp liền mạch với pipeline ASR (chuyển đổi mẫu tự động giữa 16kHz <-> 48kHz).
    """

    def __init__(
        self,
        model_name: str = "DeepFilterNet3",
        models_dir: Optional[str] = None,
        device: Optional[str] = None,
        post_filter: bool = False,
        atten_lim_db: Optional[float] = None,
        log_level: str = "WARNING"
    ):
        """
        Khởi tạo mô hình DeepFilterNet.
        :param model_name: Tên model ('DeepFilterNet3', 'DeepFilterNet2', 'DeepFilterNet')
        :param models_dir: Thư mục chứa model hoặc checkpoint đã giải nén
        :param device: 'cuda', 'cpu' hoặc None (tự động phát hiện)
        :param post_filter: Bật post-filter (tăng cường khử nhiễu nhẹ cho vùng nhiều tạp âm)
        :param atten_lim_db: Giới hạn mức độ suy giảm nhiễu (dB) nếu muốn giữ lại một phần âm nền tự nhiên
        :param log_level: Mức độ log ('INFO', 'WARNING', 'ERROR')
        """
        self.model_name = model_name
        self.atten_lim_db = atten_lim_db
        self.post_filter = post_filter

        if models_dir is None:
            models_dir = os.path.join(CURRENT_DIR, "DeepFilterNet", "models")
        self.models_dir = models_dir

        # Tự động giải nén model nếu chỉ có file .zip
        model_path = os.path.join(self.models_dir, model_name)
        zip_path = os.path.join(self.models_dir, f"{model_name}.zip")
        if not os.path.exists(model_path) and os.path.exists(zip_path):
            print(f"[DeepFilter] Đang tự động giải nén '{zip_path}'...", flush=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.models_dir)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục hoặc checkpoint của model tại '{model_path}'")

        # Thiết lập thiết bị
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Khởi tạo DeepFilterNet
        self.model, self.df_state, self.suffix, self.epoch = init_df(
            model_base_dir=model_path,
            post_filter=self.post_filter,
            log_level=log_level,
            log_file=None
        )
        self.df_sr = self.df_state.sr()  # 48000 Hz

        # Khởi động trước CUDA (Warm up)
        if self.device == "cuda":
            with torch.no_grad():
                dummy = torch.zeros((1, self.df_sr), device="cpu")
                enhance(self.model, self.df_state, dummy)

    def _filter_tensor_segmented(
        self,
        tensor_48k: torch.Tensor,
        lim: Optional[float],
        segment_s: float = 30.0,
        overlap_s: float = 1.0
    ) -> torch.Tensor:
        """
        Xử lý khử nhiễu các đoạn audio dài bằng cơ chế phân đoạn (chunking)
        kết hợp chồng lấn mượt (cross-fade) để:
        1. Tránh lỗi cuDNN error (CUDNN_STATUS_NOT_SUPPORTED) khi audio > 10 phút.
        2. Tránh tràn bộ nhớ GPU VRAM (CUDA Out of Memory) với file dài.
        3. Triệt tiêu hoàn toàn hiện tượng 'click' hay đứt gãy pha tại ranh giới chunk.
        """
        total_samples = tensor_48k.shape[-1]
        seg_samples = int(segment_s * self.df_sr)
        ovl_samples = int(overlap_s * self.df_sr)
        hop_samples = seg_samples - ovl_samples

        if total_samples <= seg_samples:
            return enhance(self.model, self.df_state, tensor_48k, pad=True, atten_lim_db=lim)

        output = torch.zeros_like(tensor_48k)
        fade_in = torch.linspace(0.0, 1.0, ovl_samples)
        fade_out = torch.linspace(1.0, 0.0, ovl_samples)

        start = 0
        first = True
        while start < total_samples:
            end = min(start + seg_samples, total_samples)
            chunk = tensor_48k[:, start:end]

            chunk_enhanced = enhance(self.model, self.df_state, chunk, pad=True, atten_lim_db=lim)
            curr_len = chunk_enhanced.shape[-1]

            if first:
                output[:, :curr_len] = chunk_enhanced
                first = False
            else:
                actual_ovl = min(ovl_samples, curr_len)
                prev_region = output[:, start : start + actual_ovl]
                curr_region = chunk_enhanced[:, :actual_ovl]
                blended = prev_region * fade_out[:actual_ovl] + curr_region * fade_in[:actual_ovl]
                output[:, start : start + actual_ovl] = blended

                if curr_len > actual_ovl:
                    output[:, start + actual_ovl : start + curr_len] = chunk_enhanced[:, actual_ovl:]

            start += hop_samples

        return output

    def filter_audio(
        self,
        audio_np: np.ndarray,
        orig_sr: int = 16000,
        target_sr: int = 16000,
        atten_lim_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Khử nhiễu trực tiếp từ mảng NumPy float32 (thường dùng trong pipeline ASR hoặc sliding window).
        :param audio_np: Mảng NumPy 1D (mono) hoặc 2D [channels, samples] với giá trị float [-1.0, 1.0]
        :param orig_sr: Tần số mẫu đầu vào (mặc định 16000 Hz của ASR)
        :param target_sr: Tần số mẫu đầu ra mong muốn (mặc định 16000 Hz cho model ASR)
        :param atten_lim_db: Ghi đè giới hạn suy hao âm nếu cần
        :return: Mảng NumPy float32 1D (mono) đã lọc sạch ở tần số target_sr
        """
        if audio_np is None or len(audio_np) == 0:
            return audio_np

        is_1d = (audio_np.ndim == 1)
        if is_1d:
            tensor = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
        else:
            tensor = torch.from_numpy(audio_np.astype(np.float32))
            if tensor.shape[0] > 1:
                # Trộn mono nếu là stereo
                tensor = tensor.mean(dim=0, keepdim=True)

        # 1. Resample lên 48kHz (tần số chuẩn của DeepFilterNet)
        if orig_sr != self.df_sr:
            tensor_48k = resample(tensor, orig_sr, self.df_sr)
        else:
            tensor_48k = tensor

        # 2. Thực hiện khử nhiễu (tự động phân đoạn an toàn với audio dài)
        lim = atten_lim_db if atten_lim_db is not None else self.atten_lim_db
        with torch.no_grad():
            enhanced_48k = self._filter_tensor_segmented(
                tensor_48k,
                lim=lim,
                segment_s=30.0,
                overlap_s=1.0
            )

        # 3. Resample về target_sr (thường là 16kHz để cấp cho model ASR)
        if target_sr != self.df_sr:
            enhanced_target = resample(enhanced_48k, self.df_sr, target_sr)
        else:
            enhanced_target = enhanced_48k

        out_np = enhanced_target.squeeze(0).cpu().numpy()
        return out_np

    def filter_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        target_sr: int = 16000
    ) -> str:
        """
        Khử nhiễu toàn bộ một file âm thanh và lưu kết quả.
        :param input_path: Đường dẫn file âm thanh đầu vào (wav, mp3, flac...)
        :param output_path: Đường dẫn lưu file sau khi lọc. Nếu None sẽ tự động đặt tên theo tiền tố `_df_filtered.wav`
        :param target_sr: Tần số mẫu của file xuất ra (mặc định 16000 Hz)
        :return: Đường dẫn file kết quả
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_df_filtered.wav"

        import soundfile as sf
        audio_data, file_sr = sf.read(input_path, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        enhanced_np = self.filter_audio(
            audio_np=audio_data,
            orig_sr=file_sr,
            target_sr=target_sr
        )

        sf.write(output_path, enhanced_np, target_sr)
        return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Kiểm tra DeepFilterNet Module...")
    filter_mod = DeepFilter()
    print(f"Khởi tạo thành công trên thiết bị: {filter_mod.device}")

    # Tạo tín hiệu thử nghiệm 2s tần số 16kHz (gồm tiếng sin 440Hz + nhiễu trắng)
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(t))
    noisy = (clean + noise).astype(np.float32)

    t0 = time.perf_counter()
    filtered = filter_mod.filter_audio(noisy, orig_sr=sr, target_sr=sr)
    t1 = time.perf_counter()

    print(f"Đã lọc 2.0s audio trong: {(t1 - t0)*1000:.2f} ms (RTF: {(t1 - t0)/2.0:.4f})")
    print(f"Kích thước đầu ra: {filtered.shape}, kiểu dữ liệu: {filtered.dtype}")
    print("=" * 60)
