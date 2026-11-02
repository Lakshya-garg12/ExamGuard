"""
ExamGuard - Model Training Pipeline
Trains custom YOLO detector on the prepared ExamGuard dataset.
Automatically logs metrics, generates loss graphs, and saves best weights to models/trained/best.pt.
"""

import os
import shutil
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.config import BASE_DIR, MODELS_DIR, RUNS_DIR

def train_model(
    data_yaml: str,
    base_model: str = "yolo11n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    project_name: str = "ExamGuard_Detection_Model"
):
    print("=" * 60)
    print(" ExamGuard Custom YOLO Training Pipeline")
    print("=" * 60)
    print(f"Dataset Config: {data_yaml}")
    print(f"Base Weights:   {base_model}")
    print(f"Epochs:         {epochs}")
    print(f"Image Size:     {imgsz}")
    print(f"Batch Size:     {batch}")
    print("=" * 60)

    # Load base pretrained model
    model = YOLO(base_model)

    # Run Training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(RUNS_DIR / "train"),
        name=project_name,
        exist_ok=True,
        save=True,
        plots=True
    )

    # Path where best.pt was saved
    train_dir = RUNS_DIR / "train" / project_name
    best_weights = train_dir / "weights" / "best.pt"

    if best_weights.exists():
        destination = MODELS_DIR / "trained" / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, destination)
        print(f"\n[+] Success! Best model weights automatically saved to: {destination}")
    else:
        print(f"\n[!] Warning: Could not locate best.pt in {train_dir / 'weights'}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExamGuard YOLO Training")
    parser.add_argument("--data", type=str, default="data/processed/ExamGuard_Unified_YOLO/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model base")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
