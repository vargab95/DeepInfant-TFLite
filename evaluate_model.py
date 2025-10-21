#!/usr/bin/env python3
"""
Model evaluation script to calculate precision, recall, F1-score and other metrics
"""

import json
from pathlib import Path
from collections import defaultdict
import random

def load_ground_truth_labels(data_dir):
    """
    Load ground truth labels from the dataset directory structure
    """
    data_dir = Path(data_dir)
    ground_truth = {}
    
    # Class mapping from directory names to label indices
    class_mapping = {
        'belly_pain': 0,
        'burping': 1, 
        'cold_hot': 2,
        'discomfort': 3,
        'hungry': 4,
        'lonely': 5,
        'scared': 6,
        'tired': 7,
        'unknown': 8
    }
    
    class_names = list(class_mapping.keys())
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            # Map directory names to standard class names
            dir_name = class_dir.name.lower()
            true_class = 'unknown'  # default
            
            if 'belly' in dir_name or 'pain' in dir_name:
                true_class = 'belly_pain'
            elif 'burp' in dir_name:
                true_class = 'burping'
            elif 'cold' in dir_name or 'hot' in dir_name:
                true_class = 'cold_hot'
            elif 'discomfort' in dir_name:
                true_class = 'discomfort'
            elif 'hungry' in dir_name:
                true_class = 'hungry'
            elif 'lonely' in dir_name:
                true_class = 'lonely'
            elif 'scared' in dir_name:
                true_class = 'scared'
            elif 'tired' in dir_name:
                true_class = 'tired'
            
            # Get all audio files in this directory
            for audio_file in class_dir.glob('*.*'):
                if audio_file.suffix.lower() in ['.wav', '.caf', '.3gp']:
                    ground_truth[audio_file.name] = class_mapping[true_class]
    
    return ground_truth, class_names

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate precision, recall, F1-score for each class and overall
    """
    num_classes = len(class_names)
    
    # Initialize confusion matrix
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label][pred_label] += 1
    
    # Calculate metrics for each class
    metrics = {}
    total_samples = len(y_true)
    
    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
        tn = total_samples - tp - fp - fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn  # Total samples for this class
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    # Calculate overall metrics
    total_tp = sum(metrics[cls]['tp'] for cls in class_names)
    total_fp = sum(metrics[cls]['fp'] for cls in class_names)
    total_fn = sum(metrics[cls]['fn'] for cls in class_names)
    
    # Macro averages (unweighted)
    macro_precision = sum(metrics[cls]['precision'] for cls in class_names) / num_classes
    macro_recall = sum(metrics[cls]['recall'] for cls in class_names) / num_classes
    macro_f1 = sum(metrics[cls]['f1_score'] for cls in class_names) / num_classes
    
    # Weighted averages (by support)
    total_support = sum(metrics[cls]['support'] for cls in class_names)
    weighted_precision = sum(metrics[cls]['precision'] * metrics[cls]['support'] for cls in class_names) / total_support if total_support > 0 else 0
    weighted_recall = sum(metrics[cls]['recall'] * metrics[cls]['support'] for cls in class_names) / total_support if total_support > 0 else 0
    weighted_f1 = sum(metrics[cls]['f1_score'] * metrics[cls]['support'] for cls in class_names) / total_support if total_support > 0 else 0
    
    # Overall accuracy
    accuracy = total_tp / total_samples if total_samples > 0 else 0
    
    overall_metrics = {
        'accuracy': accuracy,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1
        },
        'total_samples': total_samples
    }
    
    return metrics, overall_metrics, confusion_matrix

def evaluate_model(model_path='deepinfant_tensorflow.tflite', data_dir='Data/v2', sample_size=100):
    """
    Evaluate the model and calculate precision metrics
    """
    print("🎯 DeepInfant Model Evaluation")
    print("=" * 50)
    
    # Load model (using simple predictor)
    try:
        import sys
        sys.path.append('.')
        from predict_simple import SimpleTFLitePredictor
        predictor = SimpleTFLitePredictor(model_path)
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Could not import predictor: {e}")
        print("Make sure predict_simple.py exists and the model has been trained.")
        return
    
    # Load ground truth
    ground_truth, class_names = load_ground_truth_labels(data_dir)
    print(f"✅ Loaded {len(ground_truth)} ground truth labels")
    
    # Sample files for evaluation (to avoid processing all files)
    all_files = list(ground_truth.keys())
    if len(all_files) > sample_size:
        eval_files = random.sample(all_files, sample_size)
        print(f"📊 Evaluating on {sample_size} randomly sampled files")
    else:
        eval_files = all_files
        print(f"📊 Evaluating on all {len(eval_files)} files")
    
    # Get predictions
    y_true = []
    y_pred = []
    predictions_detail = []
    
    print("🔄 Running predictions...")
    
    for filename in eval_files:
        # Find the full path
        full_path = None
        for audio_file in Path(data_dir).glob(f"**/{filename}"):
            full_path = audio_file
            break
        
        if full_path:
            try:
                predicted_class, confidence, _ = predictor.predict(str(full_path))
                
                # Convert class names to indices
                true_idx = ground_truth[filename]
                pred_idx = class_names.index(predicted_class) if predicted_class in class_names else 8  # unknown
                
                y_true.append(true_idx)
                y_pred.append(pred_idx)
                
                predictions_detail.append({
                    'filename': filename,
                    'true_class': class_names[true_idx],
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct': true_idx == pred_idx
                })
                
            except Exception as e:
                print(f"❌ Error predicting {filename}: {e}")
    
    print(f"✅ Completed {len(y_true)} predictions")
    
    # Calculate metrics
    metrics, overall_metrics, confusion_matrix = calculate_metrics(y_true, y_pred, class_names)
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy: {overall_metrics['accuracy']:.3f} ({overall_metrics['accuracy']*100:.1f}%)")
    print(f"   Total samples: {overall_metrics['total_samples']}")
    
    print(f"\n📈 Macro Averages (unweighted):")
    print(f"   Precision: {overall_metrics['macro_avg']['precision']:.3f}")
    print(f"   Recall: {overall_metrics['macro_avg']['recall']:.3f}")
    print(f"   F1-Score: {overall_metrics['macro_avg']['f1_score']:.3f}")
    
    print(f"\n⚖️  Weighted Averages (by class size):")
    print(f"   Precision: {overall_metrics['weighted_avg']['precision']:.3f}")
    print(f"   Recall: {overall_metrics['weighted_avg']['recall']:.3f}")
    print(f"   F1-Score: {overall_metrics['weighted_avg']['f1_score']:.3f}")
    
    print(f"\n📋 Per-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)
    
    for class_name in class_names:
        m = metrics[class_name]
        print(f"{class_name:<12} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['support']:<8}")
    
    # Show confusion matrix
    print(f"\n🔀 Confusion Matrix:")
    print("   " + "".join(f"{i:>4}" for i in range(len(class_names))))
    for i, row in enumerate(confusion_matrix):
        print(f"{i:>2} " + "".join(f"{val:>4}" for val in row))
    
    print(f"\nClass indices:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    # Show some example predictions
    print(f"\n🔍 Example Predictions:")
    correct_preds = [p for p in predictions_detail if p['correct']][:3]
    incorrect_preds = [p for p in predictions_detail if not p['correct']][:3]
    
    if correct_preds:
        print(f"   ✅ Correct predictions:")
        for pred in correct_preds:
            print(f"      {pred['filename'][:50]}")
            print(f"      → {pred['predicted_class']} (confidence: {pred['confidence']:.1%})")
    
    if incorrect_preds:
        print(f"   ❌ Incorrect predictions:")
        for pred in incorrect_preds:
            print(f"      {pred['filename'][:50]}")
            print(f"      → Predicted: {pred['predicted_class']}, True: {pred['true_class']} (confidence: {pred['confidence']:.1%})")
    
    # Save detailed results
    results = {
        'overall_metrics': overall_metrics,
        'per_class_metrics': metrics,
        'confusion_matrix': confusion_matrix,
        'class_names': class_names,
        'predictions': predictions_detail
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: evaluation_results.json")
    
    return results

if __name__ == "__main__":
    evaluate_model()