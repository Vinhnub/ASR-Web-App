import os
import sys
import re
import json
from typing import List, Dict

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SUBDIR = os.path.join(SCRIPT_DIR, "data")
VIMEDCSS_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ViMedCSS_Test", "test_transcripts.txt"))
OUTPUT_JSONL = os.path.join(DATA_SUBDIR, "vimedcss_benchmark_test_set.jsonl")


def prepare_vimedcss(src_file: str, output_file: str, max_samples: int = 500) -> List[Dict]:
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"ViMedCSS transcript file not found at: {src_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[Dataset] Parsing ViMedCSS from: {src_file}...")

    samples = []
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            audio_id = parts[0].strip()
            keywords = parts[1].strip()
            target_text = parts[2].strip()

            # Clean up target text extra spaces
            target_text = re.sub(r'\s+', ' ', target_text)

            # Strip punctuation marks for raw ASR input simulation
            # We retain words, digits, and intra-word hyphens like 'beta-glucan'
            # Remove punctuation characters: . , ? ! : ; " ' ( ) [ ] { } ~ %
            cleaned = re.sub(r'[.,?:!;\"\'\(\)\[\]\{\}~%]', '', target_text)
            raw_words = cleaned.lower().split()
            raw_text = " ".join(raw_words).strip()

            if len(raw_words) >= 3:
                samples.append({
                    "id": len(samples) + 1,
                    "audio_id": audio_id,
                    "keywords": keywords,
                    "raw_text": raw_text,
                    "target_text": target_text,
                    "num_words": len(raw_words)
                })

            if len(samples) >= max_samples:
                break

    print(f"[Dataset] Parsed {len(samples)} valid samples from ViMedCSS.")
    total_words = sum(s["num_words"] for s in samples)
    print(f"[Dataset] Total words in selected split: {total_words} (Avg: {total_words/len(samples):.1f} words/sample)")

    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Dataset] Successfully saved benchmark dataset to: {output_file}")
    return samples


def main():
    samples = prepare_vimedcss(VIMEDCSS_SRC, OUTPUT_JSONL, max_samples=300)
    print("\n[Preview 3 Samples]:")
    for s in samples[:3]:
        print(f"  [ID #{s['id']}] Audio: {s['audio_id']} (KW: {s['keywords']})")
        print(f"    Raw    : {s['raw_text']}")
        print(f"    Target : {s['target_text']}")


if __name__ == "__main__":
    main()
