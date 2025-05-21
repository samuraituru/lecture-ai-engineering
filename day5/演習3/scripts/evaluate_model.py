#!/usr/bin/env python
import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# モデルのインポート (実際のプロジェクト構造に合わせて調整)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from day5.演習3.src.model import train_and_evaluate_model

def evaluate_model(output_file):
    """モデルを評価して結果をJSONファイルに保存"""
    print("モデルを評価しています...")
    
    # モデルの評価 (実際のプロジェクトに合わせて実装)
    metrics = train_and_evaluate_model()
    
    # 評価指標を保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"評価指標を保存しました: {output_file}")
    print(f"精度: {metrics.get('accuracy', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルを評価して指標を保存")
    parser.add_argument("--output-file", required=True, help="評価指標を保存するJSONファイルのパス")
    args = parser.parse_args()
    
    evaluate_model(args.output_file)