"""
Named Entity Recognition Dataset - Model Training Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Script to train NER models using spaCy, Transformers, or scikit-learn
"""

import json
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_spacy_data(data: List[Dict[str, Any]]) -> List[tuple]:
    """
    Convert dataset to spaCy training format.
    
    Args:
        data: List of data samples with entities
        
    Returns:
        List of (text, entities_dict) tuples for spaCy
    """
    spacy_data = []
    for sample in data:
        text = sample['text']
        entities = {}
        for entity in sample.get('entities', []):
            label = entity['label']
            start = entity['start']
            end = entity['end']
            
            if label not in entities:
                entities[label] = []
            entities[label].append((start, end))
        
        # Convert to spaCy format: list of (start, end, label) tuples
        spacy_entities = []
        for label, positions in entities.items():
            for start, end in positions:
                spacy_entities.append((start, end, label))
        
        spacy_data.append((text, {'entities': spacy_entities}))
    
    return spacy_data


def train_spacy_model(train_data: List[Dict[str, Any]], 
                     output_dir: str = 'ner_model',
                     n_iter: int = 10):
    """
    Train a spaCy NER model.
    
    Args:
        train_data: Training data in JSON format
        output_dir: Directory to save the trained model
        n_iter: Number of training iterations
    """
    try:
        import spacy
        from spacy.util import minibatch, compounding
        
        # Load a blank English model or existing model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Loading blank English model...")
            nlp = spacy.blank('en')
        
        # Add NER to pipeline if not present
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe('ner', last=True)
        else:
            ner = nlp.get_pipe('ner')
        
        # Add labels
        labels = set()
        for sample in train_data:
            for entity in sample.get('entities', []):
                labels.add(entity['label'])
        
        for label in labels:
            ner.add_label(label)
        
        # Prepare training data
        spacy_train_data = prepare_spacy_data(train_data)
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            # Initialize optimizer
            optimizer = nlp.begin_training()
            
            print(f"Training spaCy model for {n_iter} iterations...")
            for itn in range(n_iter):
                losses = {}
                batches = minibatch(spacy_train_data, size=compounding(4.0, 32.0, 1.001))
                
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, losses=losses)
                
                if itn % 2 == 0:
                    print(f"Iteration {itn}: Loss = {losses.get('ner', 0):.4f}")
        
        # Save model
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.to_disk(output_dir)
        print(f"Model saved to {output_dir}")
        
        return nlp
    
    except ImportError:
        print("spaCy is not installed. Install it with: pip install spacy")
        return None


def evaluate_model(model, test_data: List[Dict[str, Any]]):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained spaCy model
        test_data: Test data in JSON format
    """
    if model is None:
        return
    
    correct = 0
    total = 0
    
    for sample in test_data:
        text = sample['text']
        true_entities = {(e['start'], e['end'], e['label']) 
                         for e in sample.get('entities', [])}
        
        doc = model(text)
        pred_entities = {(ent.start_char, ent.end_char, ent.label_) 
                        for ent in doc.ents}
        
        correct += len(true_entities & pred_entities)
        total += len(true_entities)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy


if __name__ == '__main__':
    train_path = '../dataset/train.json'
    test_path = '../dataset/test.json'
    
    if os.path.exists(train_path):
        print("Loading training dataset...")
        train_data = load_dataset(train_path)
        print(f"Loaded {len(train_data)} training samples")
        
        print("\nTraining spaCy NER model...")
        model = train_spacy_model(train_data, n_iter=10)
        
        if os.path.exists(test_path) and model is not None:
            print("\nLoading test dataset...")
            test_data = load_dataset(test_path)
            print(f"Loaded {len(test_data)} test samples")
            
            print("\nEvaluating model...")
            evaluate_model(model, test_data)
    else:
        print(f"Training file not found: {train_path}")

