---
license: cc-by-nc-nd-4.0
---
# Vietnamese Streaming Speech-to-Text (ASR) — ZipFormer-30M-RNNT-Streaming-6000h

## 🔍 Overview
The **Vietnamese Streaming Speech-to-Text (ASR)** model is built on the **ZipFormer architecture with chunk size 16,32,64** — an improved variant of the Conformer — featuring only **30 million parameters** yet.  
On CPU, the model can transcribe a **1-second audio chunk in just 0.05 seconds**, designed for streaming-based tasks with low latency requirements.

---

## 🚀 Online Demo

You can test the streaming and none-streaming model directly here:  
👉 https://huggingface.co/spaces/hynt/k2-automatic-speech-recognition-demo

---

## ⚙️ Model Architecture and Training strategy:
- **Architecture:** ZipFormer  
- **Parameters:** ~30M  
- **Language:** Vietnamese  
- **Loss Function:** RNN-Transducer (RNNT Loss)
- **Chunk Size:** 16, 32, 64  
- **Framework:** PyTorch + k2
- **Training strategy**: Carefully preprocess the data, apply an augmentation strategy based on the distribution of out-of-vocabulary (OOV) tokens and refine the transcriptions using Whisper.  
- **Optimized for:** High-speed CPU inference  

---

## 🧠 Training Data
The model was trained on approximately **6000 hours of high-quality Vietnamese speech** collected from various public datasets:

| Dataset |  |  |
|----------|----------|----------|
| VLSP2020 | VLSP2021 | VLSP2023-voting-pseudo-labeled |
| VLSP2023 | FPT | VIET_BUD500 |
| VietSpeech | FLEURS | VietMed_Labeled |
| Sub-GigaSpeech2-Vi | ViVoice | Sub-PhoAudioBook |

---

## 🧪 Evaluation Results

| **Dataset** | **ZipFormer-30M-6000h** | **ZipFormer-30M-Streaming-chunk32-6000h** | **ChunkFormer-110M-3000h** | **PhoWhisper-Large-1.5B-800h** | **VietASR-ZipFormer-68M-70.000h** |
|--------------|--------------------------|------------------------------------|-----------------------------|--------------------------------|---------------------------------|
| **VLSP2020-Test-T1** | **12.29** | 16.68 | 14.09 | 13.75 | 14.45 |
| **VLSP2023-PublicTest** | **10.40** | 14.29 | 16.15 | 16.83 | 14.70 |
| **VLSP2023-PrivateTest** | **11.10** | 14.36 | 17.12 | 17.10 | 15.07 |
| **VLSP2025-PublicTest** | **7.97** | 12.70 | 15.55 | 16.14 | 13.55 |
| **VLSP2025-PrivateTest** | **8.10** | 12.80 | 16.07 | 16.31 | 13.97 |
| **GigaSpeech2-Test** | 7.56 | 9.72 | 10.35 | 10.00 | **6.88** |

> Lower is better (WER %)

---

## 🏆 Achievements
By training this none-streaming model architecture on 4,000 hours of data, I **won First Place** in the **Vietnamese Language Speech Processing (VLSP)** competition **2025**.
Comprehensive details about **training data**, **optimization strategies**, **architecture improvements**, and **evaluation methodologies** are available in the paper below:

👉 [Read the full paper on ACL](https://aclanthology.org/2025.vlsp-1.4.pdf)

---

## ⚡ Inference Speed

| **Device** | **Audio Length** | **Inference Time** |
|-------------|------------------|--------------------|
| CPU (Hugging Face Basic) | 1 seconds audio chunk | **0.05 s** |
| GPU (RTX 3090) | 1 seconds audio chunk | **< 0.01 s** |

---

## ⚙️ How to Run This Model

Please refer to the following guides for instructions on how to run and deploy this model:  
- **For Torch JIT Script:** [https://k2-fsa.github.io/sherpa/](https://k2-fsa.github.io/sherpa/)  
- **For ONNX:** [https://k2-fsa.github.io/sherpa/onnx/](https://k2-fsa.github.io/sherpa/onnx/)
- **For Streaming Web Test:**: [https://github.com/k2-fsa/sherpa/tree/master/sherpa/bin](https://github.com/k2-fsa/sherpa/tree/master/sherpa/bin)

## 💬 Summary
The **ZipFormer-30M-RNNT-6000h** and **ZipFormer-30M-RNNT-Streaming-6000h** model demonstrates that a lightweight architecture can still achieve state-of-the-art accuracy for Vietnamese ASR.  
It is designed for **fast deployment on CPU-based systems**, making it ideal for **real-time speech recognition**, **callbots**, and **embedded speech interfaces**.

---