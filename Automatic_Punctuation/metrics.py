import re
from typing import List, Dict, Tuple
from collections import defaultdict


def split_words_and_puncts(text: str) -> List[Tuple[str, str, str]]:
    """
    Splits text into a list of tuples: (raw_lower_word, case_type, punct)
    case_type: 'lower', 'title', 'upper'
    punct: '.', ',', '?', ':', '!', ';', ''
    """
    tokens = text.strip().split()
    results = []
    
    for tok in tokens:
        # Match trailing punctuation
        m = re.search(r'([.,?:!;]+)$', tok)
        if m:
            punct = m.group(1)[-1]  # Take the last punctuation mark if multiple
            word = tok[:m.start()]
        else:
            punct = ''
            word = tok
            
        if not word:
            continue
            
        if word.isupper() and len(word) > 1:
            case_type = 'upper'
        elif word[0].isupper():
            case_type = 'title'
        else:
            case_type = 'lower'
            
        results.append((word.lower(), case_type, punct))
        
    return results


def calculate_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes comprehensive punctuation and capitalization metrics.
    """
    total_sentences = len(references)
    exact_matches = 0
    
    total_words = 0
    correct_casing = 0
    
    # Punctuation counts: label -> {'tp': 0, 'fp': 0, 'fn': 0}
    punct_classes = ['.', ',', '?', ':', '!']
    punct_stats = {p: {'tp': 0, 'fp': 0, 'fn': 0} for p in punct_classes}
    total_punct_tp = 0
    total_punct_fp = 0
    total_punct_fn = 0
    
    total_edit_distance = 0
    total_ref_words = 0
    
    for hyp, ref in zip(hypotheses, references):
        if hyp.strip() == ref.strip():
            exact_matches += 1
            
        hyp_items = split_words_and_puncts(hyp)
        ref_items = split_words_and_puncts(ref)
        
        # Word error rate estimation
        hyp_words = hyp.strip().split()
        ref_words = ref.strip().split()
        total_ref_words += len(ref_words)
        total_edit_distance += levenshtein_distance(hyp_words, ref_words)
        
        # Alignment for word-by-word evaluation
        min_len = min(len(hyp_items), len(ref_items))
        
        for i in range(min_len):
            total_words += 1
            h_word, h_case, h_punct = hyp_items[i]
            r_word, r_case, r_punct = ref_items[i]
            
            # Casing
            if h_case == r_case:
                correct_casing += 1
                
            # Punctuation evaluation
            for p in punct_classes:
                if h_punct == p and r_punct == p:
                    punct_stats[p]['tp'] += 1
                elif h_punct == p and r_punct != p:
                    punct_stats[p]['fp'] += 1
                elif h_punct != p and r_punct == p:
                    punct_stats[p]['fn'] += 1
                    
            if h_punct and r_punct:
                if h_punct == r_punct:
                    total_punct_tp += 1
                else:
                    total_punct_fp += 1
                    total_punct_fn += 1
            elif h_punct and not r_punct:
                total_punct_fp += 1
            elif not h_punct and r_punct:
                total_punct_fn += 1
                
        # Handle length mismatch penalties
        if len(hyp_items) > min_len:
            for i in range(min_len, len(hyp_items)):
                _, _, h_punct = hyp_items[i]
                if h_punct in punct_stats:
                    punct_stats[h_punct]['fp'] += 1
                if h_punct:
                    total_punct_fp += 1
        elif len(ref_items) > min_len:
            for i in range(min_len, len(ref_items)):
                _, _, r_punct = ref_items[i]
                if r_punct in punct_stats:
                    punct_stats[r_punct]['fn'] += 1
                if r_punct:
                    total_punct_fn += 1

    # Compute F1 for each punct
    detailed_metrics = {}
    f1_sum = 0
    valid_classes = 0
    for p in punct_classes:
        tp = punct_stats[p]['tp']
        fp = punct_stats[p]['fp']
        fn = punct_stats[p]['fn']
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        detailed_metrics[f"punct_{p}_precision"] = round(precision, 2)
        detailed_metrics[f"punct_{p}_recall"] = round(recall, 2)
        detailed_metrics[f"punct_{p}_f1"] = round(f1, 2)
        if (tp + fn) > 0:
            f1_sum += f1
            valid_classes += 1

    # Overall punct micro F1
    overall_p = (total_punct_tp / (total_punct_tp + total_punct_fp)) * 100 if (total_punct_tp + total_punct_fp) > 0 else 0.0
    overall_r = (total_punct_tp / (total_punct_tp + total_punct_fn)) * 100 if (total_punct_tp + total_punct_fn) > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    casing_acc = (correct_casing / total_words) * 100 if total_words > 0 else 0.0
    exact_match_rate = (exact_matches / total_sentences) * 100 if total_sentences > 0 else 0.0
    wer = (total_edit_distance / total_ref_words) * 100 if total_ref_words > 0 else 0.0

    return {
        "sentence_count": total_sentences,
        "total_words": total_words,
        "exact_match_rate": round(exact_match_rate, 2),
        "wer": round(wer, 2),
        "casing_accuracy": round(casing_acc, 2),
        "punct_overall_precision": round(overall_p, 2),
        "punct_overall_recall": round(overall_r, 2),
        "punct_overall_f1": round(overall_f1, 2),
        "punct_macro_f1": round(f1_sum / valid_classes, 2) if valid_classes > 0 else 0.0,
        **detailed_metrics
    }


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
