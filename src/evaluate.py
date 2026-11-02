"""
ExamGuard - Model Evaluation & Performance Reporting
Computes Precision, Recall, mAP50, and mAP50-95 on the test split and produces formatted reports.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.config import DEFAULT_CUSTOM_MODEL, BASE_DIR

def evaluate_model(model_path: str, data_yaml: str, split: str = "test"):
    print("=" * 60)
    print(" ExamGuard Model Test Set Evaluation")
    print("=" * 60)
    print(f"Model:   {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Split:   {split}")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        save_json=True,
        plots=True
    )

    print("\n" + "=" * 45)
    print(" RECORDED EVALUATION METRICS:")
    print("=" * 45)
    print(f" Precision: {metrics.box.mp * 100:.2f}%")
    print(f" Recall:    {metrics.box.mr * 100:.2f}%")
    print(f" mAP50:     {metrics.box.map50 * 100:.2f}%")
    print(f" mAP50-95:  {metrics.box.map * 100:.2f}%")
    print("=" * 45)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExamGuard Model Evaluation")
    parser.add_argument("--model", type=str, default=str(DEFAULT_CUSTOM_MODEL), help="Path to best.pt")
    parser.add_argument("--data", type=str, default="data/processed/ExamGuard_Unified_YOLO/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (val/test)")
    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.split)
