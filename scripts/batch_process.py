"""
Named Entity Recognition Dataset - Batch Processing Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Advanced batch processing script for NER dataset with parallel processing and progress tracking
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import time
from collections import Counter
import pandas as pd


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single sample and extract statistics.
    
    Args:
        sample: Single data sample with text and entities
        
    Returns:
        Dictionary with processed statistics
    """
    text = sample.get('text', '')
    entities = sample.get('entities', [])
    
    return {
        'id': sample.get('id', 0),
        'text_length': len(text),
        'word_count': len(text.split()),
        'entity_count': len(entities),
        'entity_types': [e.get('label', '') for e in entities],
        'avg_entity_length': sum(len(e.get('text', '')) for e in entities) / len(entities) if entities else 0,
        'entities': entities
    }


def batch_process_dataset(data: List[Dict[str, Any]], 
                         batch_size: int = 10,
                         use_multiprocessing: bool = False,
                         max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Process dataset in batches with progress tracking.
    
    Args:
        data: List of data samples
        batch_size: Number of samples per batch
        use_multiprocessing: Use multiprocessing instead of threading
        max_workers: Maximum number of worker threads/processes
        
    Returns:
        List of processed samples
    """
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    results = []
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    with executor_class(max_workers=max_workers) as executor:
        with tqdm(total=len(data), desc="Processing samples") as pbar:
            futures = []
            for batch in batches:
                future = executor.submit(process_batch, batch)
                futures.append(future)
            
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)
                pbar.update(len(batch_results))
    
    return results


def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of samples."""
    return [process_single_sample(sample) for sample in batch]


def extract_entity_statistics(processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract comprehensive statistics from processed data.
    
    Args:
        processed_data: List of processed samples
        
    Returns:
        Dictionary with statistics
    """
    all_entity_types = []
    entity_type_counts = Counter()
    text_lengths = []
    word_counts = []
    entity_counts = []
    
    for sample in processed_data:
        text_lengths.append(sample['text_length'])
        word_counts.append(sample['word_count'])
        entity_counts.append(sample['entity_count'])
        all_entity_types.extend(sample['entity_types'])
        entity_type_counts.update(sample['entity_types'])
    
    return {
        'total_samples': len(processed_data),
        'total_entities': sum(entity_counts),
        'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
        'avg_entities_per_sample': sum(entity_counts) / len(entity_counts) if entity_counts else 0,
        'entity_type_distribution': dict(entity_type_counts),
        'min_text_length': min(text_lengths) if text_lengths else 0,
        'max_text_length': max(text_lengths) if text_lengths else 0,
        'min_entities': min(entity_counts) if entity_counts else 0,
        'max_entities': max(entity_counts) if entity_counts else 0
    }


def export_statistics(stats: Dict[str, Any], output_file: str):
    """Export statistics to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics exported to {output_file}")


def export_processed_data(processed_data: List[Dict[str, Any]], output_file: str, format: str = 'json'):
    """
    Export processed data to file.
    
    Args:
        processed_data: List of processed samples
        output_file: Output file path
        format: Export format ('json' or 'csv')
    """
    if format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    elif format == 'csv':
        rows = []
        for sample in processed_data:
            for entity in sample.get('entities', []):
                rows.append({
                    'id': sample['id'],
                    'text_length': sample['text_length'],
                    'word_count': sample['word_count'],
                    'entity_text': entity.get('text', ''),
                    'entity_type': entity.get('label', ''),
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    print(f"Processed data exported to {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process NER dataset')
    parser.add_argument('--input', type=str, default='../dataset/train.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='processed_data.json', help='Output file')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--multiprocessing', action='store_true', help='Use multiprocessing')
    parser.add_argument('--stats', type=str, default='statistics.json', help='Statistics output file')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    data = load_dataset(args.input)
    print(f"Loaded {len(data)} samples")
    
    print("\nProcessing dataset...")
    start_time = time.time()
    processed_data = batch_process_dataset(
        data, 
        batch_size=args.batch_size,
        use_multiprocessing=args.multiprocessing,
        max_workers=args.workers
    )
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    
    print("\nExtracting statistics...")
    stats = extract_entity_statistics(processed_data)
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Average text length: {stats['avg_text_length']:.2f} characters")
    print(f"  Average word count: {stats['avg_word_count']:.2f} words")
    print(f"  Average entities per sample: {stats['avg_entities_per_sample']:.2f}")
    print(f"\nEntity Type Distribution:")
    for entity_type, count in sorted(stats['entity_type_distribution'].items()):
        print(f"  {entity_type}: {count}")
    
    print(f"\nExporting processed data to {args.output}...")
    export_processed_data(processed_data, args.output, format=args.format)
    
    print(f"Exporting statistics to {args.stats}...")
    export_statistics(stats, args.stats)
    
    print("\nBatch processing complete!")

