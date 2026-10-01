# -*- coding: utf-8 -*-
"""
Shortcut module để chạy benchmark ViMedCSS trực tiếp từ thư mục Speaker_Diarization.
"""
import os
import sys

# Chuyển tiếp thực thi sang benchmark_sortformer_vimedcss.py ở thư mục gốc ASR
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from benchmark_sortformer_vimedcss import main

if __name__ == "__main__":
    main()
