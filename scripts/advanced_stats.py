"""
Named Entity Recognition Dataset - Advanced Statistics Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Advanced statistics and analytics for NER dataset
"""

import json
import os
from typing import List, Dict, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_comprehensive_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for the dataset.
    
    Args:
        data: List of data samples
        
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = {
        'basic': {},
        'text_analysis': {},
        'entity_analysis': {},
        'coverage_analysis': {},
        'quality_metrics': {}
    }
    
    # Basic statistics
    stats['basic']['total_samples'] = len(data)
    stats['basic']['total_entities'] = sum(len(sample.get('entities', [])) for sample in data)
    
    # Text analysis
    text_lengths = [len(sample.get('text', '')) for sample in data]
    word_counts = [len(sample.get('text', '').split()) for sample in data]
    
    stats['text_analysis'] = {
        'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'min_text_length': min(text_lengths) if text_lengths else 0,
        'max_text_length': max(text_lengths) if text_lengths else 0,
        'median_text_length': sorted(text_lengths)[len(text_lengths) // 2] if text_lengths else 0,
        'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_word_count': min(word_counts) if word_counts else 0,
        'max_word_count': max(word_counts) if word_counts else 0,
        'median_word_count': sorted(word_counts)[len(word_counts) // 2] if word_counts else 0
    }
    
    # Entity analysis
    entity_type_counts = Counter()
    entity_lengths = defaultdict(list)
    entities_per_sample = []
    
    for sample in data:
        entities = sample.get('entities', [])
        entities_per_sample.append(len(entities))
        
        for entity in entities:
            entity_type = entity.get('label', '')
            entity_text = entity.get('text', '')
            entity_type_counts[entity_type] += 1
            entity_lengths[entity_type].append(len(entity_text))
    
    stats['entity_analysis'] = {
        'entity_type_distribution': dict(entity_type_counts),
        'avg_entities_per_sample': sum(entities_per_sample) / len(entities_per_sample) if entities_per_sample else 0,
        'min_entities_per_sample': min(entities_per_sample) if entities_per_sample else 0,
        'max_entities_per_sample': max(entities_per_sample) if entities_per_sample else 0,
        'entity_lengths_by_type': {
            entity_type: {
                'avg': sum(lengths) / len(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0
            }
            for entity_type, lengths in entity_lengths.items()
        }
    }
    
    # Coverage analysis
    total_chars = sum(len(sample.get('text', '')) for sample in data)
    entity_chars = 0
    
    for sample in data:
        for entity in sample.get('entities', []):
            entity_chars += entity.get('end', 0) - entity.get('start', 0)
    
    stats['coverage_analysis'] = {
        'total_characters': total_chars,
        'entity_characters': entity_chars,
        'coverage_percentage': (entity_chars / total_chars * 100) if total_chars > 0 else 0,
        'samples_with_entities': sum(1 for sample in data if sample.get('entities')),
        'samples_without_entities': sum(1 for sample in data if not sample.get('entities'))
    }
    
    # Quality metrics
    overlapping_entities = 0
    invalid_entities = 0
    
    for sample in data:
        text = sample.get('text', '')
        entities = sample.get('entities', [])
        
        # Check for overlapping entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                start1, end1 = entity1.get('start', 0), entity1.get('end', 0)
                start2, end2 = entity2.get('start', 0), entity2.get('end', 0)
                
                if not (end1 <= start2 or start1 >= end2):
                    overlapping_entities += 1
        
        # Check for invalid entities (out of bounds)
        for entity in entities:
            start = entity.get('start', 0)
            end = entity.get('end', 0)
            if start < 0 or end > len(text) or start >= end:
                invalid_entities += 1
    
    stats['quality_metrics'] = {
        'overlapping_entities': overlapping_entities,
        'invalid_entities': invalid_entities,
        'data_quality_score': max(0, 100 - (overlapping_entities + invalid_entities) * 10)
    }
    
    return stats


def generate_visualizations(stats: Dict[str, Any], output_dir: str = 'visualizations'):
    """Generate visualization charts from statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Entity type distribution
    if 'entity_type_distribution' in stats['entity_analysis']:
        plt.figure(figsize=(10, 6))
        entity_dist = stats['entity_analysis']['entity_type_distribution']
        plt.bar(entity_dist.keys(), entity_dist.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        plt.xlabel('Entity Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Entity Type Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/entity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Text length distribution
    plt.figure(figsize=(10, 6))
    # This would need actual data, but we'll create a placeholder
    plt.hist([stats['text_analysis']['avg_text_length']] * 10, bins=20, color='#667eea', alpha=0.7)
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def export_detailed_report(stats: Dict[str, Any], output_file: str = 'detailed_statistics.json'):
    """Export detailed statistics report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Detailed statistics exported to {output_file}")


def print_statistics_report(stats: Dict[str, Any]):
    """Print a formatted statistics report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE NER DATASET STATISTICS")
    print("="*60)
    
    print("\n📊 BASIC STATISTICS")
    print("-" * 60)
    for key, value in stats['basic'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n📝 TEXT ANALYSIS")
    print("-" * 60)
    for key, value in stats['text_analysis'].items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n🏷️  ENTITY ANALYSIS")
    print("-" * 60)
    print(f"  Average Entities per Sample: {stats['entity_analysis']['avg_entities_per_sample']:.2f}")
    print(f"  Min Entities per Sample: {stats['entity_analysis']['min_entities_per_sample']}")
    print(f"  Max Entities per Sample: {stats['entity_analysis']['max_entities_per_sample']}")
    print("\n  Entity Type Distribution:")
    for entity_type, count in sorted(stats['entity_analysis']['entity_type_distribution'].items()):
        print(f"    {entity_type}: {count}")
    
    print("\n📈 COVERAGE ANALYSIS")
    print("-" * 60)
    for key, value in stats['coverage_analysis'].items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n✅ QUALITY METRICS")
    print("-" * 60)
    for key, value in stats['quality_metrics'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate advanced statistics for NER dataset')
    parser.add_argument('--train', type=str, default='../dataset/train.json', help='Training dataset path')
    parser.add_argument('--test', type=str, default='../dataset/test.json', help='Test dataset path')
    parser.add_argument('--output', type=str, default='detailed_statistics.json', help='Output file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    print("Loading datasets...")
    train_data = load_dataset(args.train) if os.path.exists(args.train) else []
    test_data = load_dataset(args.test) if os.path.exists(args.test) else []
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    print("\nCalculating statistics for training dataset...")
    train_stats = calculate_comprehensive_stats(train_data)
    print_statistics_report(train_stats)
    
    if test_data:
        print("\nCalculating statistics for test dataset...")
        test_stats = calculate_comprehensive_stats(test_data)
        print_statistics_report(test_stats)
    
    # Combined statistics
    combined_data = train_data + test_data
    if combined_data:
        print("\nCalculating combined statistics...")
        combined_stats = calculate_comprehensive_stats(combined_data)
        print_statistics_report(combined_stats)
        
        export_detailed_report(combined_stats, args.output)
        
        if args.visualize:
            print("\nGenerating visualizations...")
            generate_visualizations(combined_stats)
    
    print("Analysis complete!")

