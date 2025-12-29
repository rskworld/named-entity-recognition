"""
Named Entity Recognition Dataset - Model Evaluation Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Advanced model evaluation and comparison tools
"""

import json
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entities_from_sample(sample: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    """Extract entities as (start, end, label) tuples."""
    entities = []
    for entity in sample.get('entities', []):
        entities.append((
            entity.get('start', 0),
            entity.get('end', 0),
            entity.get('label', '')
        ))
    return entities


def calculate_metrics(true_entities: List[Tuple[int, int, str]], 
                      pred_entities: List[Tuple[int, int, str]]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        true_entities: List of (start, end, label) tuples for ground truth
        pred_entities: List of (start, end, label) tuples for predictions
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    
    # True Positives: entities that are in both sets
    tp = len(true_set & pred_set)
    
    # False Positives: predicted entities not in true set
    fp = len(pred_set - true_set)
    
    # False Negatives: true entities not in predicted set
    fn = len(true_set - pred_set)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def evaluate_model_predictions(true_data: List[Dict[str, Any]], 
                              pred_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate model predictions against ground truth.
    
    Args:
        true_data: Ground truth data
        pred_data: Predicted data (same structure as true_data)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(true_data) != len(pred_data):
        raise ValueError("True and predicted data must have the same length")
    
    overall_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    per_entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    sample_metrics = []
    
    for true_sample, pred_sample in zip(true_data, pred_data):
        true_entities = extract_entities_from_sample(true_sample)
        pred_entities = extract_entities_from_sample(pred_sample)
        
        # Overall metrics
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        overall_metrics['all']['tp'] += len(true_set & pred_set)
        overall_metrics['all']['fp'] += len(pred_set - true_set)
        overall_metrics['all']['fn'] += len(true_set - pred_set)
        
        # Per-entity-type metrics
        true_by_type = defaultdict(set)
        pred_by_type = defaultdict(set)
        
        for start, end, label in true_entities:
            true_by_type[label].add((start, end, label))
        
        for start, end, label in pred_entities:
            pred_by_type[label].add((start, end, label))
        
        all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
        for entity_type in all_types:
            true_type_set = true_by_type[entity_type]
            pred_type_set = pred_by_type[entity_type]
            
            per_entity_metrics[entity_type]['tp'] += len(true_type_set & pred_type_set)
            per_entity_metrics[entity_type]['fp'] += len(pred_type_set - true_type_set)
            per_entity_metrics[entity_type]['fn'] += len(true_type_set - pred_type_set)
        
        # Sample-level metrics
        sample_metrics.append(calculate_metrics(true_entities, pred_entities))
    
    # Calculate overall metrics
    overall = overall_metrics['all']
    overall_precision = overall['tp'] / (overall['tp'] + overall['fp']) if (overall['tp'] + overall['fp']) > 0 else 0.0
    overall_recall = overall['tp'] / (overall['tp'] + overall['fn']) if (overall['tp'] + overall['fn']) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Calculate per-entity metrics
    per_entity_results = {}
    for entity_type, metrics in per_entity_metrics.items():
        precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0
        recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_entity_results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn']
        }
    
    # Sample-level statistics
    avg_precision = np.mean([m['precision'] for m in sample_metrics])
    avg_recall = np.mean([m['recall'] for m in sample_metrics])
    avg_f1 = np.mean([m['f1'] for m in sample_metrics])
    
    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': overall['tp'],
            'fp': overall['fp'],
            'fn': overall['fn']
        },
        'per_entity': per_entity_results,
        'sample_level': {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'std_precision': np.std([m['precision'] for m in sample_metrics]),
            'std_recall': np.std([m['recall'] for m in sample_metrics]),
            'std_f1': np.std([m['f1'] for m in sample_metrics])
        }
    }


def print_evaluation_report(results: Dict[str, Any]):
    """Print formatted evaluation report."""
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)
    
    print("\n📊 OVERALL METRICS")
    print("-" * 70)
    overall = results['overall']
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1 Score:  {overall['f1']:.4f}")
    print(f"\n  True Positives:  {overall['tp']}")
    print(f"  False Positives: {overall['fp']}")
    print(f"  False Negatives: {overall['fn']}")
    
    print("\n🏷️  PER-ENTITY METRICS")
    print("-" * 70)
    print(f"{'Entity Type':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 70)
    for entity_type, metrics in sorted(results['per_entity'].items()):
        print(f"{entity_type:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print("\n📈 SAMPLE-LEVEL STATISTICS")
    print("-" * 70)
    sample = results['sample_level']
    print(f"  Average Precision: {sample['avg_precision']:.4f} (±{sample['std_precision']:.4f})")
    print(f"  Average Recall:    {sample['avg_recall']:.4f} (±{sample['std_recall']:.4f})")
    print(f"  Average F1 Score:  {sample['avg_f1']:.4f} (±{sample['std_f1']:.4f})")
    
    print("\n" + "="*70 + "\n")


def export_evaluation_results(results: Dict[str, Any], output_file: str):
    """Export evaluation results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results exported to {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate NER model predictions')
    parser.add_argument('--true', type=str, required=True, help='Ground truth JSON file')
    parser.add_argument('--pred', type=str, required=True, help='Predictions JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file')
    
    args = parser.parse_args()
    
    print("Loading datasets...")
    true_data = load_dataset(args.true)
    pred_data = load_dataset(args.pred)
    
    print(f"Loaded {len(true_data)} ground truth samples")
    print(f"Loaded {len(pred_data)} prediction samples")
    
    print("\nEvaluating model...")
    results = evaluate_model_predictions(true_data, pred_data)
    
    print_evaluation_report(results)
    export_evaluation_results(results, args.output)
    
    print("Evaluation complete!")

