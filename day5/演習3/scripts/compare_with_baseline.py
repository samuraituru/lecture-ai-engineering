#!/usr/bin/env python
import argparse
import json
import os

def compare_with_baseline(current_file, baseline_file, output_file):
    """現在のメトリクスとベースラインを比較"""
    print(f"比較: 現在={current_file}, ベースライン={baseline_file}")
    
    # 現在のメトリクスを読み込み
    try:
        with open(current_file, 'r') as f:
            current = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 現在のメトリクスファイルが見つかりません: {current_file}")
        return False
    
    # ベースラインを読み込み (存在しない場合は現在のメトリクスをベースラインとする)
    baseline_exists = os.path.exists(baseline_file)
    if baseline_exists:
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
        except:
            print(f"警告: ベースラインファイルが読み込めません: {baseline_file}")
            baseline = current
            baseline_exists = False
    else:
        print(f"警告: ベースラインファイルが存在しません: {baseline_file}")
        baseline = current
    
    # 比較結果を格納
    results = []
    has_degradation = False
    
    # 評価指標を比較
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in current and metric in baseline:
            current_value = current[metric]
            baseline_value = baseline[metric]
            
            # 性能低下の閾値 (例: 1%以上の低下で警告)
            threshold = 0.01
            
            if current_value < baseline_value - threshold:
                status = "FAIL"
                has_degradation = True
            elif current_value >= baseline_value:
                status = "IMPROVED"
            else:
                status = "ACCEPTABLE"  # わずかな低下は許容
                
            results.append({
                "metric": metric, 
                "current": current_value,
                "baseline": baseline_value,
                "diff": current_value - baseline_value,
                "status": status
            })
    
    # 推論時間の比較 (存在する場合)
    if 'inference_time' in current and 'inference_time' in baseline:
        current_time = current['inference_time']
        baseline_time = baseline['inference_time']
        
        # 時間増加の閾値 (例: 10%以上の増加で警告)
        threshold = 0.1
        
        if current_time > baseline_time * (1 + threshold):
            status = "FAIL"
            has_degradation = True
        else:
            status = "PASS"
            
        results.append({
            "metric": "inference_time", 
            "current": current_time,
            "baseline": baseline_time,
            "diff": current_time - baseline_time,
            "status": status
        })
    
    # テキスト形式のレポートを作成
    with open(output_file, 'w') as f:
        f.write("モデル性能比較レポート\n")
        f.write("=====================\n\n")
        
        if not baseline_exists:
            f.write("注意: ベースラインが存在しないため、現在のメトリクスが新しいベースラインとなります。\n\n")
        
        f.write("メトリクス比較:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'メトリクス':<15}{'現在値':<10}{'ベースライン':<10}{'差分':<10}{'ステータス':<10}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['metric']:<15}{result['current']:<10.4f}{result['baseline']:<10.4f}{result['diff']:<10.4f}{result['status']:<10}\n")
        
        f.write("\n全体判定: ")
        if has_degradation:
            f.write("FAIL - 性能の劣化が検出されました\n")
        else:
            f.write("PASS - 性能の劣化はありません\n")
    
    # JSON形式でも保存
    with open(output_file.replace('.txt', '.json'), 'w') as f:
        json.dump({
            "has_degradation": has_degradation,
            "baseline_exists": baseline_exists,
            "results": results
        }, f, indent=2)
    
    print(f"比較結果を保存しました: {output_file}")
    return not has_degradation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="現在のモデル性能とベースラインを比較")
    parser.add_argument("--current", required=True, help="現在のメトリクスファイル")
    parser.add_argument("--baseline", required=True, help="ベースラインメトリクスファイル")
    parser.add_argument("--output", required=True, help="比較結果の出力ファイル")
    args = parser.parse_args()
    
    success = compare_with_baseline(args.current, args.baseline, args.output)
    if not success:
        exit(1)