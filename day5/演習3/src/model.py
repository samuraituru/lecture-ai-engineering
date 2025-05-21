import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def load_data():
    """データを読み込む（実際のプロジェクトに合わせて実装）"""
    # 例としてirisデータセットを使用
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def preprocess_data(X):
    """データの前処理を行う"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def train_model(X, y, random_state=42):
    """モデルを訓練する"""
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    return model

def evaluate_predictions(y_true, y_pred, inference_time=None):
    """予測結果を評価する"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='macro')),
        "recall": float(recall_score(y_true, y_pred, average='macro')),
        "f1_score": float(f1_score(y_true, y_pred, average='macro'))
    }
    
    if inference_time is not None:
        metrics["inference_time"] = float(inference_time)
        
    return metrics

def train_and_evaluate_model():
    """モデルを訓練して評価する"""
    # データの読み込み
    X, y = load_data()
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 前処理
    X_train_processed = preprocess_data(X_train)
    X_test_processed = preprocess_data(X_test)
    
    # モデル訓練
    model = train_model(X_train_processed, y_train)
    
    # 推論時間の計測
    start_time = time.time()
    y_pred = model.predict(X_test_processed)
    inference_time = time.time() - start_time
    
    # 評価
    metrics = evaluate_predictions(y_test, y_pred, inference_time)
    
    return metrics