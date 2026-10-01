import os
import sys
import json
import urllib.request
from typing import List, Dict

sys.stdout.reconfigure(encoding='utf-8')

DATA_URL = "https://raw.githubusercontent.com/ductho9799/JointCapPunc/master/data/test.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SUBDIR = os.path.join(DATA_DIR, "data")
RAW_TEST_FILE = os.path.join(DATA_SUBDIR, "vicappunc_test_raw.txt")
PROCESSED_TEST_FILE = os.path.join(DATA_SUBDIR, "benchmark_test_set.jsonl")


def download_raw_test_set(url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
        print(f"[Dataset] Raw test file already exists at: {output_path} ({os.path.getsize(output_path)} bytes)")
        return
    print(f"[Dataset] Downloading ViCapPunc test set from {url}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, output_path)
    print(f"[Dataset] Download completed: {os.path.getsize(output_path)} bytes.")


def tag_to_char(punct_tag: str) -> str:
    mapping = {
        'O': '',
        'PERIOD': '.',
        'COMMA': ',',
        'QUESTION': '?',
        'QMARK': '?',
        'COLON': ':',
        'EXCLAM': '!',
        'SEMICOLON': ';',
        'HYPHEN': '-',
        'ELLIPSIS': '...',
    }
    return mapping.get(punct_tag.upper(), '')


def format_word(word: str, case_tag: str) -> str:
    word = word.strip()
    if not word:
        return ""
    if case_tag == '1':
        return word.capitalize()
    elif case_tag == '2':
        return word.upper()
    else:
        return word.lower()


def parse_vicappunc_file(file_path: str, max_samples: int = 1000) -> List[Dict]:
    print(f"[Dataset] Parsing {file_path}...")
    samples = []
    
    current_raw_words = []
    current_target_words = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_raw_words:
                    raw_text = " ".join(current_raw_words).strip()
                    target_text = " ".join(current_target_words).strip()
                    # Clean spacing before punctuation marks
                    for p in [".", ",", "?", "!", ":", ";"]:
                        target_text = target_text.replace(f" {p}", p)
                    if len(current_raw_words) >= 3:
                        samples.append({
                            "id": len(samples) + 1,
                            "raw_text": raw_text,
                            "target_text": target_text,
                            "num_words": len(current_raw_words)
                        })
                        if len(samples) >= max_samples:
                            break
                    current_raw_words = []
                    current_target_words = []
                continue
                
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
                
            word, case_tag, punct_tag = parts[0], parts[1], parts[2]
            p_char = tag_to_char(punct_tag)
            c_word = format_word(word, case_tag)
            
            current_raw_words.append(word.lower())
            if p_char:
                current_target_words.append(f"{c_word}{p_char}")
            else:
                current_target_words.append(c_word)
                
            # Sentence delimiter
            if punct_tag.upper() in ['PERIOD', 'QUESTION', 'QMARK', 'EXCLAM']:
                raw_text = " ".join(current_raw_words).strip()
                target_text = " ".join(current_target_words).strip()
                for p in [".", ",", "?", "!", ":", ";"]:
                    target_text = target_text.replace(f" {p}", p)
                if len(current_raw_words) >= 3:
                    samples.append({
                        "id": len(samples) + 1,
                        "raw_text": raw_text,
                        "target_text": target_text,
                        "num_words": len(current_raw_words)
                    })
                    if len(samples) >= max_samples:
                        break
                current_raw_words = []
                current_target_words = []
                
    if current_raw_words and len(samples) < max_samples:
        raw_text = " ".join(current_raw_words).strip()
        target_text = " ".join(current_target_words).strip()
        for p in [".", ",", "?", "!", ":", ";"]:
            target_text = target_text.replace(f" {p}", p)
        if len(current_raw_words) >= 3:
            samples.append({
                "id": len(samples) + 1,
                "raw_text": raw_text,
                "target_text": target_text,
                "num_words": len(current_raw_words)
            })

    print(f"[Dataset] Parsed {len(samples)} clean sentences.")
    return samples


def main():
    download_raw_test_set(DATA_URL, RAW_TEST_FILE)
    samples = parse_vicappunc_file(RAW_TEST_FILE, max_samples=500)
    
    with open(PROCESSED_TEST_FILE, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"[Dataset] Saved {len(samples)} benchmark samples to {PROCESSED_TEST_FILE}")
    total_words = sum(s["num_words"] for s in samples)
    print(f"[Dataset] Total words in benchmark split: {total_words}")
    print("[Dataset] Sample preview:")
    for s in samples[:3]:
        print(f"  [Sample #{s['id']}]")
        print(f"    Raw    : {s['raw_text']}")
        print(f"    Target : {s['target_text']}")


if __name__ == "__main__":
    main()
