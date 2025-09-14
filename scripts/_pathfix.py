# scripts/_pathfix.py
from pathlib import Path
import sys

def add_project_root():
    root = Path(__file__).resolve().parents[1]  # 项目根目录
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))