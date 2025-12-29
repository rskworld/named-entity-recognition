"""
Named Entity Recognition Dataset - Load Dataset Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Script to load and parse NER dataset from JSON and CSV formats
"""

import json
import pandas as pd
import os
from typing import List, Dict, Any


def load_ner_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load NER dataset from JSON file.
    
    Args:
        file_path: Path to the JSON dataset file
        
    Returns:
        List of dictionaries containing text and entities
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_ner_csv(file_path: str) -> pd.DataFrame:
    """
    Load NER dataset from CSV file.
    
    Args:
        file_path: Path to the CSV dataset file
        
    Returns:
        DataFrame with columns: text, entity_type, start, end, entity_text
    """
    return pd.read_csv(file_path)


def get_entity_statistics(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get statistics about entity types in the dataset.
    
    Args:
        data: List of data samples with entities
        
    Returns:
        Dictionary with entity type counts
    """
    entity_counts = {}
    for sample in data:
        for entity in sample.get('entities', []):
            label = entity.get('label', '')
            entity_counts[label] = entity_counts.get(label, 0) + 1
    return entity_counts


def convert_to_bio_format(data: List[Dict[str, Any]]) -> List[List[tuple]]:
    """
    Convert dataset to BIO tagging format.
    
    Args:
        data: List of data samples with entities
        
    Returns:
        List of sentences, each as a list of (word, tag) tuples
    """
    bio_sentences = []
    
    for sample in data:
        text = sample['text']
        entities = sample.get('entities', [])
        
        # Create a list of (char_pos, label) for each entity
        entity_tags = {}
        for entity in entities:
            start = entity['start']
            end = entity['end']
            label = entity['label']
            for pos in range(start, end):
                if pos == start:
                    entity_tags[pos] = f'B-{label}'
                else:
                    entity_tags[pos] = f'I-{label}'
        
        # Tokenize and assign tags
        words = text.split()
        bio_sentence = []
        char_pos = 0
        
        for word in words:
            word_start = char_pos
            word_end = char_pos + len(word)
            
            # Check if word starts an entity
            if word_start in entity_tags:
                tag = entity_tags[word_start]
            elif any(pos in entity_tags for pos in range(word_start, word_end)):
                # Word is inside an entity
                tag = entity_tags.get(word_start, 'O')
                if tag == 'O':
                    # Find the first tag in this word
                    for pos in range(word_start, word_end):
                        if pos in entity_tags:
                            tag = entity_tags[pos]
                            break
            else:
                tag = 'O'
            
            bio_sentence.append((word, tag))
            char_pos = word_end + 1  # +1 for space
        
        bio_sentences.append(bio_sentence)
    
    return bio_sentences


if __name__ == '__main__':
    # Example usage
    train_path = '../dataset/train.json'
    test_path = '../dataset/test.json'
    
    if os.path.exists(train_path):
        print("Loading training dataset...")
        train_data = load_ner_dataset(train_path)
        print(f"Loaded {len(train_data)} training samples")
        
        stats = get_entity_statistics(train_data)
        print("\nEntity Statistics:")
        for entity_type, count in sorted(stats.items()):
            print(f"  {entity_type}: {count}")
        
        print("\nFirst sample:")
        print(json.dumps(train_data[0], indent=2))
    
    if os.path.exists(test_path):
        print("\nLoading test dataset...")
        test_data = load_ner_dataset(test_path)
        print(f"Loaded {len(test_data)} test samples")

