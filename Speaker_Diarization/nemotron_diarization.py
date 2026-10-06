import os
import time
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from Speaker_Diarization.streaming_sortformer import SpeakerSegment

class NemotronDiarizer:
    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-3-Diarization",
        device: Optional[str] = None,
        collar: float = 0.25,
        min_duration: float = 0.20,
    ):
        self.model_name = model_name
        self.collar = collar
        self.min_duration = min_duration
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[NemotronDiarizer] Thiết bị thực thi: {self.device.upper()}")
        self._load_model()
        
    def _load_model(self):
        print(f"[NemotronDiarizer] Đang tải mô hình '{self.model_name}'...")
        start_t = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        elapsed = time.time() - start_t
        print(f"[NemotronDiarizer] Tải mô hình thành công trong {elapsed:.2f}s!")

    def _postprocess_segments(self, segments: List[SpeakerSegment], collar: float, min_duration: float) -> List[SpeakerSegment]:
        if not segments:
            return []
            
        sorted_segs = sorted(segments, key=lambda s: (s.speaker, s.start))
        merged_by_speaker = {}
        for seg in sorted_segs:
            spk = seg.speaker
            if spk not in merged_by_speaker:
                merged_by_speaker[spk] = [seg]
            else:
                last_seg = merged_by_speaker[spk][-1]
                if seg.start <= last_seg.end + collar:
                    merged_by_speaker[spk][-1] = SpeakerSegment(
                        start=last_seg.start,
                        end=max(last_seg.end, seg.end),
                        speaker=spk,
                        confidence=(last_seg.confidence + seg.confidence) / 2.0
                    )
                else:
                    merged_by_speaker[spk].append(seg)
                    
        all_merged = []
        for spk, seg_list in merged_by_speaker.items():
            for s in seg_list:
                if s.duration >= min_duration:
                    all_merged.append(s)
                    
        all_merged.sort(key=lambda s: s.start)
        return all_merged

    def diarize_numpy(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> List[SpeakerSegment]:
        if len(audio_pcm) == 0:
            return []
            
        target_sr = self.processor.feature_extractor.sampling_rate
        audio_input = audio_pcm
        if sample_rate != target_sr:
            # We assume it's 16k which is what model uses
            pass
            
        inputs = self.processor(
            audio_input, 
            sampling_rate=target_sr, 
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)
        
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            
        extracted = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        
        segments = []
        for seg in extracted:
            segments.append(SpeakerSegment(
                start=seg["Start"],
                end=seg["End"],
                speaker=f"speaker_{seg['Speaker']}",
                confidence=1.0
            ))
            
        c = self.collar if collar is None else collar
        m = self.min_duration if min_duration is None else min_duration
        return self._postprocess_segments(segments, c, m)

    def detect_vad_activity(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        silence_threshold_s: float = 0.6,
        speech_prob_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Phân tích hoạt động giọng nói (VAD) và phát hiện trạng thái dứt lượt phát biểu.
        """
        if len(audio_pcm) == 0:
            return {
                "is_speaking": False,
                "silence_duration_s": 0.0,
                "last_speech_end": 0.0,
                "segments": [],
                "active_speaker": None
            }

        target_sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(
            audio_pcm, 
            sampling_rate=target_sr, 
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        total_duration = len(audio_pcm) / sample_rate
        probs = torch.sigmoid(logits[0])
        frame_max_prob = torch.max(probs, dim=-1).values.cpu().numpy()

        num_frames = len(frame_max_prob)
        frame_dur = total_duration / num_frames if num_frames > 0 else 0.01

        check_frames = max(1, int(silence_threshold_s / frame_dur))
        tail_frames = frame_max_prob[-check_frames:] if check_frames < num_frames else frame_max_prob
        is_speaking = bool(np.any(tail_frames >= speech_prob_threshold))

        speech_indices = np.where(frame_max_prob >= speech_prob_threshold)[0]
        if len(speech_indices) > 0:
            last_speech_idx = speech_indices[-1]
            last_speech_end = min(total_duration, float((last_speech_idx + 1) * frame_dur))
            silence_duration_s = max(0.0, total_duration - last_speech_end)
        else:
            last_speech_end = 0.0
            silence_duration_s = total_duration

        extracted = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        segments = []
        for seg in extracted:
            segments.append(SpeakerSegment(
                start=seg["Start"],
                end=seg["End"],
                speaker=f"speaker_{seg['Speaker']}",
                confidence=1.0
            ))
        c = self.collar
        m = self.min_duration
        segments = self._postprocess_segments(segments, c, m)
        active_speaker = segments[-1].speaker if segments else None

        return {
            "is_speaking": is_speaking,
            "silence_duration_s": round(silence_duration_s, 3),
            "last_speech_end": round(last_speech_end, 3),
            "segments": segments,
            "active_speaker": active_speaker
        }

    def diarize_file(
        self,
        audio_path: str,
        collar: Optional[float] = None,
        min_duration: Optional[float] = None,
        verbose: bool = True,
    ) -> List[SpeakerSegment]:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.diarize_numpy(audio, sr, collar, min_duration)

