import os
import sys
import re
from typing import List, Union, Tuple
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')


class ViBERTOnnxRunner:
    """
    ONNX Runtime runner for ViBERT-capu (supports INT8 and FP32, CPU and CUDA).
    Uses Seq2Labels GECToR edit transformations.
    """
    def __init__(self, variant: str = "int8", device: str = "cpu", chunk_size: int = 48, overlap_size: int = 12):
        self.variant = variant
        self.device = device
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.stride = chunk_size - overlap_size
        
        print(f"[ViBERTOnnxRunner] Initializing ViBERT-Capu ONNX ({variant.upper()}) on {device.upper()}...")
        local_dir = snapshot_download("welcomyou/vibert-capu-onnx")
        self.local_dir = local_dir
        
        onnx_file = "vibert-capu.int8.onnx" if variant == "int8" else "vibert-capu.onnx"
        self.model_path = os.path.join(local_dir, onnx_file)
        self.file_size_mb = round(os.path.getsize(self.model_path) / (1024 * 1024), 2)
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        # Filter available providers
        available_providers = ort.get_available_providers()
        selected_providers = [p for p in providers if p in available_providers]
        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]
            
        self.session = ort.InferenceSession(self.model_path, providers=selected_providers)
        
        # Load tokenizer
        self.start_token = "$START"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "FPTAI/vibert-base-cased",
            do_basic_tokenize=False,
            do_lower_case=False
        )
        self.tokenizer.add_tokens([self.start_token])
        
        # Load vocabulary labels
        labels_file = os.path.join(local_dir, "vocabulary", "labels.txt")
        with open(labels_file, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f]
            
        # Dynamically import utils from downloaded snapshot
        if local_dir not in sys.path:
            sys.path.append(local_dir)
        from utils import get_target_sent_by_edits
        self.get_target_sent_by_edits = get_target_sent_by_edits
        print(f"[ViBERTOnnxRunner] Loaded {onnx_file} ({self.file_size_mb} MB) successfully with {selected_providers[0]}.")

    def _predict_chunk(self, words: List[str]) -> str:
        if not words:
            return ""
        tokens = [self.start_token] + words
        enc = self.tokenizer(
            [tokens],
            is_split_into_words=True,
            return_tensors="np",
            add_special_tokens=False
        )
        word_ids = enc.word_ids(batch_index=0)
        offsets = [0]
        for i in range(1, len(word_ids)):
            if word_ids[i] != word_ids[i - 1]:
                offsets.append(i)
        input_offsets = np.array([offsets], dtype=np.int64)
        
        ort_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc["token_type_ids"].astype(np.int64),
            "input_offsets": input_offsets,
        }
        
        logits, _ = self.session.run(None, ort_inputs)
        probs = np.max(np.exp(logits[0]) / np.sum(np.exp(logits[0]), axis=-1, keepdims=True), axis=-1)
        pred_idx = np.argmax(logits[0], axis=-1)
        
        edits = []
        for i, (tok, idx, p) in enumerate(zip(tokens, pred_idx, probs)):
            sugg = self.labels[idx]
            if sugg in ["$KEEP", "@@PADDING@@", "@@UNKNOWN@@"]:
                continue
            if sugg.startswith("$TRANSFORM_") or sugg == "$DELETE":
                start_pos, end_pos = i, i + 1
            elif sugg.startswith("$APPEND_") or sugg.startswith("$MERGE_"):
                start_pos, end_pos = i + 1, i + 1
            else:
                start_pos, end_pos = i, i + 1
                
            if sugg.startswith("$TRANSFORM_") or sugg.startswith("$MERGE_"):
                clear = sugg
            elif "_" in sugg:
                clear = sugg.split("_", 1)[1]
            else:
                clear = sugg
            edits.append((start_pos - 1, end_pos - 1, clear, p))
            
        res_tokens = self.get_target_sent_by_edits(words, edits)
        out_text = " ".join(res_tokens)
        out_text = re.sub(r'\s+([,.:?])', r'\1', out_text)
        return out_text

    def predict(self, text: str) -> str:
        words = text.strip().split()
        if not words:
            return ""
            
        # ViBERT-capu xử lý tối ưu với các câu/đoạn lên đến 150 từ trong 1 forward pass
        max_chunk = 150
        if len(words) <= max_chunk:
            return self._predict_chunk(words)
            
        # Chia các đoạn dài thành các khối liên tiếp không chồng chéo để tránh lặp từ
        punctuated_chunks = []
        for i in range(0, len(words), max_chunk):
            chunk = words[i:i + max_chunk]
            if not chunk:
                break
            punctuated_chunks.append(self._predict_chunk(chunk))
            
        return " ".join(punctuated_chunks)


class VFastPunctRunner:
    """
    Runner for VFastPunct mLstmPunctCap (BiLSTM + CRF architecture).
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"[VFastPunctRunner] Initializing VFastPunct (mLstmPunctCap) on {device.upper()}...")
        
        import torch
        # Monkey patch torch.load for PyTorch 2.6+ compatibility
        orig_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return orig_load(*args, **kwargs)
        torch.load = safe_load
        
        orig_load_sd = torch.nn.Module.load_state_dict
        def safe_load_sd(self, state_dict, strict=True, assign=False):
            return orig_load_sd(self, state_dict, strict=False, assign=assign)
        torch.nn.Module.load_state_dict = safe_load_sd
        
        import vfastpunct.predictor
        vfastpunct.predictor.BASE_PATH = "C:/tmp"
        
        no_cuda = (device == "cpu") or not torch.cuda.is_available()
        self.model = vfastpunct.predictor.VFastPunct("mLstmPunctCap", no_cuda=no_cuda)
        
        model_file = "C:/tmp/mlstmpunctcap_11tsglNoAJzpCwAkWXs8OGIo2WJCqPw59.pt"
        self.file_size_mb = round(os.path.getsize(model_file) / (1024 * 1024), 2) if os.path.exists(model_file) else 397.0
        print(f"[VFastPunctRunner] Loaded mLstmPunctCap ({self.file_size_mb} MB) successfully.")

    def predict(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        try:
            return self.model(text)
        except Exception as e:
            return text
