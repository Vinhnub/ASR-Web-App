# -*- coding: utf-8 -*-
"""
================================================================================
NVIDIA TitaNet - Speaker Recognition, Verification & Enrollment Database
================================================================================
Module cung cấp khả năng trích xuất Voiceprint (Speaker Embedding), quản lý cơ sở
dữ liệu hồ sơ giọng nói (Speaker Profile Database) và nhận diện danh tính người nói
sử dụng mô hình NVIDIA TitaNet (TitaNet-Large / 1D Depthwise Separable Convolutions).

Mô hình mặc định: nvidia/speakerverification_en_titanet_large
Hỗ trợ:
- Trích xuất vector đặc trưng âm học chuẩn hóa L2 (192 hoặc 512 chiều).
- Hỗ trợ đầu vào đa dạng: Đường dẫn file WAV hoặc mảng NumPy / PyTorch Tensor trong bộ nhớ.
- Quản lý cơ sở dữ liệu giọng nói bền vững (Lưu trữ nhị phân .npz và metadata .json).
- Hỗ trợ gom nhiều mẫu âm thanh cho cùng 1 người (Centroid Embedding Averaging).
- So khớp Cosine Similarity với ngưỡng linh hoạt (Threshold Verification & Identification).
"""

import os
import sys
import time
import json
import tempfile
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

# Thiết lập UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
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

# Import load_audio từ audio_utils
try:
    from audio_utils import load_audio
except ImportError:
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_path not in sys.path:
        sys.path.append(parent_path)
    try:
        from audio_utils import load_audio
    except ImportError:
        def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                except ImportError:
                    pass
            return audio.astype(np.float32)




class TitaNetSpeakerRecognizer:
    """
    Trình trích xuất đặc trưng giọng nói (Voiceprint) và xác thực người nói
    dựa trên mô hình NVIDIA TitaNet.
    """

    def __init__(
        self,
        model_name: str = "nvidia/speakerverification_en_titanet_large",
        device: Optional[str] = None,
    ):
        """
        Khởi tạo mô hình TitaNet.

        Tham số:
            model_name: Tên checkpoint trên HuggingFace (hoặc file .nemo cục bộ)
            device: 'cuda' hoặc 'cpu' (tự động chọn nếu None)
        """
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[TitaNet] Thiết bị thực thi: {self.device.upper()}")
        self.model = self._load_model()

    def _load_model(self):
        """
        Tải mô hình TitaNet từ NeMo Toolkit.
        """
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError as e:
            raise ImportError(
                "\n[LỖI] Chưa cài đặt thư viện NVIDIA NeMo Toolkit!\n"
                "Vui lòng cài đặt bằng lệnh:\n"
                "    pip install nemo_toolkit['asr']\n"
                "Hoặc tham khảo hướng dẫn tại: https://github.com/NVIDIA/NeMo\n"
            ) from e

        print(f"[TitaNet] Đang tải mô hình '{self.model_name}'...")
        t0 = time.time()

        try:
            if os.path.isfile(self.model_name) and self.model_name.endswith(".nemo"):
                model = EncDecSpeakerLabelModel.restore_from(restore_path=self.model_name)
            else:
                model = EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name)
        except Exception as err:
            # Fallback sang checkpoint titanet_large nếu có vấn đề kết nối
            fallback_name = "titanet_large"
            if self.model_name != fallback_name:
                print(f"[Cảnh báo] Thử fallback sang '{fallback_name}'...")
                model = EncDecSpeakerLabelModel.from_pretrained(model_name=fallback_name)
                self.model_name = fallback_name
            else:
                raise err

        model = model.to(self.device)
        model.eval()

        elapsed = time.time() - t0
        print(f"[TitaNet] Tải mô hình thành công trong {elapsed:.2f}s!")
        return model

    def extract_embedding(
        self,
        audio_input: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Trích xuất vector embedding L2-normalized từ âm thanh.
        Tự động xử lý đa kênh (stereo -> mono), chuẩn hóa tần số 16kHz và đưa trực tiếp vào GPU.

        Tham số:
            audio_input: Đường dẫn file WAV hoặc mảng NumPy float32 [-1.0, 1.0]
            sample_rate: Tần số lấy mẫu (khuyến nghị chuẩn 16000 Hz)

        Trả về:
            np.ndarray: Vector 1 chiều (192 phần tử) đã chuẩn hóa L2 norm = 1.0
        """
        # 1. Chuẩn hóa âm thanh đầu vào thành mảng float32 mono 16kHz
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Không tìm thấy file âm thanh: '{audio_input}'")
            audio_array = load_audio(audio_input, target_sr=sample_rate)
        elif isinstance(audio_input, torch.Tensor):
            audio_array = audio_input.detach().cpu().numpy().squeeze()
        else:
            audio_array = np.squeeze(audio_input)

        # Đảm bảo mảng 1D mono float32
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=tuple(range(1, audio_array.ndim)))
        audio_array = audio_array.astype(np.float32)

        # Bắt buộc chuẩn hóa tần số lấy mẫu về 16000 Hz cho TitaNet
        if sample_rate != 16000:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            except Exception:
                try:
                    import scipy.signal
                    num_samples = int(len(audio_array) * 16000 / sample_rate)
                    audio_array = scipy.signal.resample(audio_array, num_samples).astype(np.float32)
                except Exception:
                    pass
            sample_rate = 16000

        # Kiểm tra thời lượng tối thiểu (~0.3 giây chuẩn 16kHz)
        min_samples = int(0.3 * sample_rate)
        if len(audio_array) < min_samples:
            pad_len = min_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, pad_len), mode="constant")

        # 2. Đưa tensor vào model forward trực tiếp trên GPU
        signal = torch.from_numpy(audio_array).unsqueeze(0).to(self.device)
        signal_len = torch.tensor([signal.shape[1]], dtype=torch.int64, device=self.device)

        with torch.no_grad():
            outputs = self.model.forward(input_signal=signal, input_signal_length=signal_len)
            if isinstance(outputs, (tuple, list)):
                emb_tensor = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                emb_tensor = outputs

            if isinstance(emb_tensor, torch.Tensor):
                emb = emb_tensor.detach().cpu().numpy().squeeze()
            else:
                emb = np.squeeze(np.array(emb_tensor))

        # 3. Chuẩn hóa L2-norm (|v| = 1)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        return emb.astype(np.float32)

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Tính độ tương đồng Cosine giữa 2 vector embedding đã chuẩn hóa L2:
        similarity = dot(emb1, emb2)
        Kết quả trong khoảng [-1.0, 1.0], giá trị càng gần 1.0 thì càng giống cùng 1 người.
        """
        emb1 = emb1.squeeze()
        emb2 = emb2.squeeze()
        sim = float(np.dot(emb1, emb2))
        return max(-1.0, min(1.0, sim))

    def verify_pair(
        self,
        audio1: Union[str, np.ndarray],
        audio2: Union[str, np.ndarray],
        threshold: float = 0.70,
        sample_rate: int = 16000,
    ) -> Tuple[bool, float]:
        """
        Xác thực xem 2 đoạn âm thanh có cùng thuộc về 1 người nói hay không.

        Trả về:
            (is_same_speaker: bool, similarity_score: float)
        """
        emb1 = self.extract_embedding(audio1, sample_rate=sample_rate)
        emb2 = self.extract_embedding(audio2, sample_rate=sample_rate)
        score = self.compute_similarity(emb1, emb2)
        is_match = score >= threshold
        return is_match, score


class SpeakerDatabase:
    """
    Quản lý kho lưu trữ hồ sơ giọng nói (Speaker Profiles) bền vững.
    Lưu trữ các vector đặc trưng trong file .npz và metadata trong file .json.
    """

    def __init__(self, db_dir: str = "./speaker_db", recognizer: Optional[TitaNetSpeakerRecognizer] = None):
        """
        Khởi tạo cơ sở dữ liệu người nói.

        Tham số:
            db_dir: Thư mục chứa dữ liệu
            recognizer: Đối tượng TitaNetSpeakerRecognizer dùng để trích xuất vector
        """
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

        self.embeddings_path = os.path.join(self.db_dir, "speakers_embeddings.npz")
        self.metadata_path = os.path.join(self.db_dir, "speakers_metadata.json")

        self.recognizer = recognizer
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}

        self.load()

    def _ensure_recognizer(self):
        if self.recognizer is None:
            self.recognizer = TitaNetSpeakerRecognizer()

    def enroll(
        self,
        name: str,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
        description: str = "",
    ) -> Tuple[np.ndarray, int]:
        """
        Đăng ký một người nói mới hoặc bổ sung mẫu giọng cho người nói đã có.

        Nếu người nói đã tồn tại trong database, vector đại diện mới sẽ là
        trung bình có trọng số (Running Centroid) của các mẫu trước và mẫu mới:
        v_new = normalize( (v_old * count + v_sample) / (count + 1) )

        Trả về:
            (current_embedding, total_sample_count)
        """
        self._ensure_recognizer()
        name = name.strip()
        if not name:
            raise ValueError("Tên người nói không được để trống!")

        print(f"[SpeakerDB] Đang trích xuất Voiceprint cho '{name}'...")
        new_emb = self.recognizer.extract_embedding(audio_input, sample_rate=sample_rate)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if name in self.embeddings:
            # Cập nhật thêm mẫu cho người cũ
            old_emb = self.embeddings[name]
            count = self.metadata[name].get("sample_count", 1)
            merged = (old_emb * count + new_emb) / (count + 1)
            norm = np.linalg.norm(merged)
            if norm > 1e-8:
                merged = merged / norm

            self.embeddings[name] = merged.astype(np.float32)
            self.metadata[name]["sample_count"] = count + 1
            self.metadata[name]["updated_at"] = now_str
            if description:
                self.metadata[name]["description"] = description

            print(f"[SpeakerDB] Đã cập nhật mẫu thứ {count + 1} cho '{name}'.")
        else:
            # Người mới
            self.embeddings[name] = new_emb.astype(np.float32)
            self.metadata[name] = {
                "name": name,
                "sample_count": 1,
                "created_at": now_str,
                "updated_at": now_str,
                "description": description,
            }
            print(f"[SpeakerDB] Đã đăng ký thành công người nói mới: '{name}'!")

        self.save()
        return self.embeddings[name], self.metadata[name]["sample_count"]

    def identify(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.70,
    ) -> Tuple[str, float]:
        """
        So sánh một vector embedding truy vấn với toàn bộ danh sách người nói trong Database.

        Trả về:
            (speaker_name, best_score)
            Nếu best_score < threshold, speaker_name sẽ là "Unknown".
        """
        if len(self.embeddings) == 0:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, enrolled_emb in self.embeddings.items():
            score = float(np.dot(query_embedding, enrolled_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            return "Unknown", max(0.0, best_score)

        return best_name, max(0.0, best_score)

    def identify_audio(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
        threshold: float = 0.70,
    ) -> Tuple[str, float]:
        """
        Nhận diện trực tiếp từ file âm thanh hoặc mảng numpy.
        """
        self._ensure_recognizer()
        emb = self.recognizer.extract_embedding(audio_input, sample_rate=sample_rate)
        return self.identify(emb, threshold=threshold)

    def delete(self, name: str) -> bool:
        """
        Xóa một người nói khỏi cơ sở dữ liệu.
        """
        removed = False
        if name in self.embeddings:
            del self.embeddings[name]
            removed = True
        if name in self.metadata:
            del self.metadata[name]
            removed = True

        if removed:
            self.save()
            print(f"[SpeakerDB] Đã xóa người nói '{name}'.")
            return True
        print(f"[SpeakerDB] Không tìm thấy '{name}' để xóa.")
        return False

    def delete_speaker(self, name: str) -> bool:
        """
        Alias cho delete().
        """
        return self.delete(name)

    def list_speakers(self) -> List[Dict]:
        """
        Liệt kê thông tin toàn bộ người nói đã đăng ký.
        """
        info_list = []
        for name, meta in self.metadata.items():
            info = dict(meta)
            info["has_embedding"] = name in self.embeddings
            info_list.append(info)
        return info_list

    def save(self):
        """
        Lưu toàn bộ vector và metadata ra đĩa.
        """
        # Lưu các mảng vector sang file npz
        np.savez_compressed(self.embeddings_path, **self.embeddings)

        # Lưu metadata sang file JSON
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        """
        Nạp dữ liệu từ đĩa nếu đã tồn tại.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[SpeakerDB] Lỗi nạp metadata: {e}")
                self.metadata = {}

        if os.path.exists(self.embeddings_path):
            try:
                data = np.load(self.embeddings_path)
                self.embeddings = {k: data[k] for k in data.files}
                print(f"[SpeakerDB] Đã nạp thành công {len(self.embeddings)} hồ sơ giọng nói từ '{self.db_dir}'.")
            except Exception as e:
                print(f"[SpeakerDB] Lỗi nạp embeddings: {e}")
                self.embeddings = {}


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA TitaNet - Quản lý Voiceprint & Nhận diện danh tính người nói"
    )
    parser.add_argument("--db-dir", type=str, default="./speaker_db", help="Thư mục lưu cơ sở dữ liệu giọng nói")
    parser.add_argument("--model", type=str, default="nvidia/speakerverification_en_titanet_large", help="Tên model TitaNet")
    parser.add_argument("--device", type=str, default=None, help="'cuda' hoặc 'cpu'")

    # Các thao tác chính
    parser.add_argument("--enroll", action="store_true", help="Đăng ký giọng nói mới")
    parser.add_argument("--name", type=str, default="", help="Tên người nói cần đăng ký")
    parser.add_argument("--audio", type=str, default="", help="File âm thanh mẫu để đăng ký hoặc nhận diện")
    parser.add_argument("--desc", type=str, default="", help="Mô tả / vai trò của người nói (tùy chọn)")

    parser.add_argument("--identify", action="store_true", help="Nhận diện người nói trong file âm thanh")
    parser.add_argument("--threshold", type=float, default=0.70, help="Ngưỡng Cosine Similarity (mặc định 0.70)")

    parser.add_argument("--verify", action="store_true", help="Xác thực xem 2 file có cùng 1 người nói không")
    parser.add_argument("--audio1", type=str, default="", help="File âm thanh thứ 1")
    parser.add_argument("--audio2", type=str, default="", help="File âm thanh thứ 2")

    parser.add_argument("--list", action="store_true", help="Xem danh sách toàn bộ người nói đã đăng ký")
    parser.add_argument("--delete", type=str, default=None, help="Xóa một người nói khỏi cơ sở dữ liệu")

    args = parser.parse_args()

    # 1. Thao tác xem danh sách
    if args.list:
        db = SpeakerDatabase(db_dir=args.db_dir)
        speakers = db.list_speakers()
        print("\n" + "=" * 70)
        print(f"DANH SÁCH NGƯỜI NÓI TRONG DATABASE ({len(speakers)} người):")
        print("-" * 70)
        print(f"{'TÊN NGƯỜI NÓI':<20} | {'SỐ MẪU':<8} | {'NGÀY TẠO':<20} | {'MÔ TẢ'}")
        print("-" * 70)
        for spk in speakers:
            print(f"{spk['name']:<20} | {spk.get('sample_count', 1):<8} | {spk.get('created_at', ''):<20} | {spk.get('description', '')}")
        print("=" * 70 + "\n")
        return

    # 2. Thao tác xóa
    if args.delete:
        db = SpeakerDatabase(db_dir=args.db_dir)
        db.delete(args.delete)
        return

    # Khởi tạo recognizer cho các tác vụ cần mô hình AI
    recognizer = TitaNetSpeakerRecognizer(model_name=args.model, device=args.device)
    db = SpeakerDatabase(db_dir=args.db_dir, recognizer=recognizer)

    # 3. Thao tác đăng ký
    if args.enroll:
        if not args.name or not args.audio:
            print("[LỖI] Đăng ký yêu cầu cả --name và --audio!")
            return
        db.enroll(name=args.name, audio_input=args.audio, description=args.desc)

    # 4. Thao tác nhận diện
    elif args.identify:
        if not args.audio:
            print("[LỖI] Nhận diện yêu cầu tham số --audio!")
            return
        spk_name, score = db.identify_audio(args.audio, threshold=args.threshold)
        print("\n" + "=" * 50)
        print(f"KẾT QUẢ NHẬN DIỆN: {args.audio}")
        print(f"  • Danh tính : {spk_name.upper()}")
        print(f"  • Độ tương đồng (Similarity): {score:.4f} (Ngưỡng: {args.threshold})")
        print("=" * 50 + "\n")

    # 5. Thao tác xác thực 2 file
    elif args.verify:
        if not args.audio1 or not args.audio2:
            print("[LỖI] Xác thực yêu cầu cả --audio1 và --audio2!")
            return
        is_same, score = recognizer.verify_pair(args.audio1, args.audio2, threshold=args.threshold)
        print("\n" + "=" * 50)
        print(f"KẾT QUẢ XÁC THỰC CẶP FILE:")
        print(f"  • File 1: {args.audio1}")
        print(f"  • File 2: {args.audio2}")
        print(f"  • Cùng 1 người?: {'ĐÚNG (CÙNG NGƯỜI)' if is_same else 'SAI (NGƯỜI KHÁC NHAU)'}")
        print(f"  • Điểm tương đồng: {score:.4f} (Ngưỡng: {args.threshold})")
        print("=" * 50 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
