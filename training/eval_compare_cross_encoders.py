#!/usr/bin/env python3
"""
Interactive cross-encoder model comparison.

Usage:
  python -m training.eval_compare_cross_encoders
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.evaluate_similarity_pipeline import compute_metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    records = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def list_available_models() -> list[Path]:
    """List all models from models/ directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    # Get ALL subdirectories
    models = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    return models


def run_crossencoder(model_path: str, cv_texts: list[str], jd_texts: list[str], batch_size: int = 32) -> tuple[list[float], str]:
    """Score pairs using cross-encoder or bi-encoder. Returns (scores, model_type)."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    model_name = Path(model_path).name
    print(f"  📦 Loading: {model_name}")

    # Try cross-encoder first
    if "cross-encoder" in model_name.lower():
        try:
            model = CrossEncoder(model_path, max_length=512)
            print(f"  🔄 Scoring {len(cv_texts)} pairs (cross-encoder)…")
            scores = model.predict(
                list(zip(cv_texts, jd_texts)),
                batch_size=batch_size,
                show_progress_bar=True,
            )
            result = []
            for score in scores:
                if isinstance(score, (list, np.ndarray)):
                    score = float(score[0]) if len(score) > 0 else 0.0
                else:
                    score = float(score)
                result.append(score)
            return result, "cross-encoder"
        except Exception as e:
            print(f"  ⚠️  CrossEncoder loading failed: {e}")

    # Fall back to bi-encoder
    try:
        model = SentenceTransformer(model_path)
        print(f"  🔄 Computing embeddings and similarity (bi-encoder)…")

        # Encode texts
        cv_embeddings = model.encode(cv_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        jd_embeddings = model.encode(jd_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

        # Compute cosine similarity
        scores = [float(np.dot(cv, jd)) for cv, jd in zip(cv_embeddings, jd_embeddings)]

        # Clamp to 0-1 range
        scores = [max(0.0, min(1.0, s)) for s in scores]

        return scores, "bi-encoder"
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        raise


def format_comparison_table(results: list[tuple[str, str, float, float, float]]) -> None:
    """Print comparison table for multiple models."""
    print("\n" + "=" * 125)
    print("📊 COMPARISON RESULTS (on identical test set)")
    print("=" * 125 + "\n")

    header = f"{'#':<3} {'Model':<40} {'Type':<15} {'MAE':<15} {'RMSE':<15} {'LabelAcc':<15}"
    sep = "─" * 125

    print(header)
    print(sep)

    for i, (name, model_type, mae, rmse, lacc) in enumerate(results, 1):
        type_label = "📍 Cross-Enc" if model_type == "cross-encoder" else "📌 Bi-Enc"
        print(f"{i:<3} {name:<40} {type_label:<15} {mae:<15.4f} {rmse:<15.4f} {lacc*100:<14.2f}%")

    print(sep)

    # Find best by LabelAcc
    best_idx = max(range(len(results)), key=lambda i: results[i][4])
    best_name, best_type, _, _, best_lacc = results[best_idx]
    print(f"\n🏆 Best model: #{best_idx + 1} {best_name} ({best_type})")

    # Show deltas vs best
    if len(results) > 1:
        _, _, best_mae, best_rmse, _ = results[best_idx]
        print(f"\n📈 Delta vs best:\n")
        for i, (name, model_type, mae, rmse, lacc) in enumerate(results, 1):
            type_icon = "📍" if model_type == "cross-encoder" else "📌"
            if i == best_idx + 1:
                print(f"   #{i} {name:<40} [{type_icon}] (baseline)")
            else:
                delta_mae = mae - best_mae
                delta_rmse = rmse - best_rmse
                delta_lacc = (lacc - results[best_idx][4]) * 100
                print(f"   #{i} {name:<40} [{type_icon}] MAE: {delta_mae:+.4f}  RMSE: {delta_rmse:+.4f}  LabelAcc: {delta_lacc:+.2f} pp")

    print("\n" + "=" * 110 + "\n")


def select_models(available_models: list[Path], count: int) -> list[Path]:
    """Interactive model selection."""
    print(f"\n📋 Available models ({len(available_models)} total):\n")

    for i, model_path in enumerate(available_models, 1):
        print(f"  {i:2d}. {model_path.name}")

    selected = []
    for slot in range(1, count + 1):
        while True:
            try:
                choice = int(input(f"\n🔹 Model {slot}/{count}: Select number (1-{len(available_models)}): ").strip())
                if 1 <= choice <= len(available_models):
                    selected_model = available_models[choice - 1]
                    if selected_model in selected:
                        print("   ⚠️  Already selected! Choose another.")
                        continue
                    selected.append(selected_model)
                    print(f"   ✅ Selected: {selected_model.name}")
                    break
                else:
                    print(f"   ❌ Invalid choice. Enter 1-{len(available_models)}")
            except ValueError:
                print("   ❌ Invalid input. Enter a number.")

    return selected


def main() -> None:
    print("\n" + "=" * 110)
    print("🔍 CROSS-ENCODER MODEL COMPARISON")
    print("=" * 110)

    # Check available models
    available_models = list_available_models()
    if not available_models:
        print("❌ No cross-encoder models found in models/ directory")
        return

    print(f"✅ Found {len(available_models)} models\n")

    # Ask how many models to compare
    max_count = len(available_models)
    while True:
        try:
            count = int(input(f"🔹 How many models to compare? (2-{max_count}): ").strip())
            if 2 <= count <= max_count:
                break
            else:
                print(f"   ❌ Enter a number between 2 and {max_count}")
        except ValueError:
            print("   ❌ Invalid input. Enter a number.")

    # Select models
    selected_models = select_models(available_models, count)

    # Load test set
    data_dir = Path("datasets/versions/v0.5/cross_encoder")
    if not data_dir.exists():
        print(f"❌ Dataset not found: {data_dir}")
        return

    print(f"\n📂 Loading test set from {data_dir.name}…")
    test_data = load_jsonl(data_dir / "cross_encoder_test.jsonl")

    cv_texts = [item["cv_text"] for item in test_data]
    jd_texts = [item["jd_text"] for item in test_data]
    # Use 'label' which is 0-1, or 'score' if label doesn't exist
    true_scores = [item.get("label", item.get("score", 0.0)) for item in test_data]

    print(f"✅ Loaded {len(test_data)} test pairs\n")

    # Evaluate each model
    results = []
    for i, model_path in enumerate(selected_models, 1):
        print(f"\n🚀 [{i}/{len(selected_models)}] Evaluating {model_path.name}…")
        scores, model_type = run_crossencoder(str(model_path), cv_texts, jd_texts, batch_size=32)
        metrics = compute_metrics(scores, true_scores)
        results.append((model_path.name, model_type, metrics.mae, metrics.rmse, metrics.label_accuracy))
        type_icon = "📍" if model_type == "cross-encoder" else "📌"
        print(f"  ✅ [{type_icon} {model_type}] MAE={metrics.mae:.4f}  RMSE={metrics.rmse:.4f}  LabelAcc={metrics.label_accuracy:.4f}")

    # Print comparison
    format_comparison_table(results)

    # Ask to save report
    while True:
        save = input("💾 Save report to JSON? (y/n): ").strip().lower()
        if save in ("y", "yes"):
            output_path = Path("artifacts/reports/comparison_cross_encoders.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = {
                "comparison": [
                    {
                        "rank": i,
                        "name": name,
                        "type": model_type,
                        "metrics": {
                            "mae": mae,
                            "rmse": rmse,
                            "label_accuracy": lacc,
                        }
                    }
                    for i, (name, model_type, mae, rmse, lacc) in enumerate(results, 1)
                ],
                "dataset": "v0.5",
                "test_pairs": len(test_data),
                "best_model": results[best_idx][0],
                "best_model_type": results[best_idx][1],
            }
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"✅ Report saved → {output_path}\n")
            break
        elif save in ("n", "no"):
            print("⏭️  Skipped report save\n")
            break
        else:
            print("❌ Enter 'y' or 'n'")


if __name__ == "__main__":
    main()
