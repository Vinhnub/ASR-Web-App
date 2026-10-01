# -*- coding: utf-8 -*-
"""
Realtime ASR Pipeline Package with Modular Toggleable Components.
Models included:
- Denoise: DeepFilterNet3
- ASR: ZipFormer (Streaming with Attention & CNN Cache via sherpa-onnx)
- Speaker Diarization: NVIDIA Streaming Sortformer (FastConformer + AOSC)
- Speaker Verification: NVIDIA TitaNet
- ASR Post-correction: Hot word Dictionary
- Automatic Punctuation: ViBERT-capu
"""
__version__ = "1.0.0"
