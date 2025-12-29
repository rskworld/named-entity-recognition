"""
Named Entity Recognition Dataset - Flask API Server
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: RESTful API server for NER dataset operations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from typing import List, Dict, Any
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load datasets
TRAIN_DATA = None
TEST_DATA = None


def load_datasets():
    """Load training and test datasets."""
    global TRAIN_DATA, TEST_DATA
    try:
        # Try relative path from scripts directory first
        train_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'train.json')
        test_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test.json')
        train_path = os.path.normpath(train_path)
        test_path = os.path.normpath(test_path)
        
        # Try alternative paths
        if not os.path.exists(train_path):
            train_path = 'dataset/train.json'
        if not os.path.exists(test_path):
            test_path = 'dataset/test.json'
        
        with open(train_path, 'r', encoding='utf-8') as f:
            TRAIN_DATA = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            TEST_DATA = json.load(f)
        print("Datasets loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: Dataset files not found: {e}. Some endpoints may not work.")
    except Exception as e:
        print(f"Error loading datasets: {e}")


# Simple NER patterns for demo
NER_PATTERNS = {
    'PERSON': [
        re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'),
        re.compile(r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b')
    ],
    'ORG': [
        re.compile(r'\b([A-Z][a-z]+ (?:Inc\.|Corporation|Corp\.|LLC|Ltd\.|Company|Co\.))\b'),
        re.compile(r'\b([A-Z]{2,})\b'),
        re.compile(r'\b(Apple|Microsoft|Google|Amazon|Facebook|Twitter|Tesla|SpaceX|Netflix|Uber|Airbnb|Spotify|LinkedIn|Zoom|Dropbox|Salesforce|Oracle|Intel|Adobe|NVIDIA|OpenAI|Meta|Alphabet)\b')
    ],
    'LOC': [
        re.compile(r'\b([A-Z][a-z]+(?:, [A-Z][a-z]+)?)\b'),
        re.compile(r'\b(New York|Los Angeles|San Francisco|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|Seattle|Denver|Washington|Boston|El Paso|Detroit|Nashville|Portland|Oklahoma City|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Kansas City|Mesa|Atlanta|Omaha|Colorado Springs|Raleigh|Virginia Beach|Miami|Oakland|Minneapolis|Tulsa|Cleveland|Wichita|Arlington)\b'),
        re.compile(r'\b(United States|USA|UK|Canada|Mexico|China|Japan|India|Germany|France|Italy|Spain|Brazil|Australia|Russia|South Korea|Netherlands|Sweden|Switzerland|Belgium|Norway|Denmark|Finland|Poland|Austria|Greece|Portugal|Ireland|Czech Republic|Romania|Hungary)\b')
    ],
    'DATE': [
        re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'),
        re.compile(r'\b\d{4}\b'),
        re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b')
    ],
    'MONEY': [
        re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'),
        re.compile(r'\b\d+(?:,\d{3})* (?:dollars|USD)\b', re.IGNORECASE)
    ],
    'PERCENT': [
        re.compile(r'\b\d+(?:\.\d+)?%\b')
    ]
}


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entities from text using pattern matching."""
    entities = []
    entity_map = {}
    
    for entity_type, patterns in NER_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                entity_text = match.group(0)
                
                # Check for overlaps
                overlap = False
                for (existing_start, existing_end), _ in entity_map.items():
                    if not (end <= existing_start or start >= existing_end):
                        overlap = True
                        break
                
                if not overlap:
                    entities.append({
                        'text': entity_text,
                        'label': entity_type,
                        'start': start,
                        'end': end
                    })
                    entity_map[(start, end)] = entity_type
    
    # Sort by start position
    entities.sort(key=lambda x: x['start'])
    return entities


@app.route('/')
def index():
    """API information endpoint."""
    return jsonify({
        'name': 'NER Dataset API',
        'version': '1.0.0',
        'author': 'RSK World',
        'website': 'https://rskworld.in',
        'email': 'help@rskworld.in',
        'phone': '+91 93305 39277',
        'endpoints': {
            '/api/extract': 'POST - Extract entities from text',
            '/api/dataset/train': 'GET - Get training dataset',
            '/api/dataset/test': 'GET - Get test dataset',
            '/api/dataset/stats': 'GET - Get dataset statistics',
            '/api/dataset/sample': 'GET - Get random sample',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'datasets_loaded': TRAIN_DATA is not None and TEST_DATA is not None
    })


@app.route('/api/extract', methods=['POST'])
def extract():
    """Extract entities from provided text."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        entities = extract_entities(text)
        
        return jsonify({
            'text': text,
            'entities': entities,
            'entity_count': len(entities),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/train', methods=['GET'])
def get_train_dataset():
    """Get training dataset."""
    if TRAIN_DATA is None:
        return jsonify({'error': 'Training dataset not loaded'}), 500
    
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', type=int, default=0)
    
    data = TRAIN_DATA[offset:]
    if limit:
        data = data[:limit]
    
    return jsonify({
        'total': len(TRAIN_DATA),
        'returned': len(data),
        'offset': offset,
        'data': data
    })


@app.route('/api/dataset/test', methods=['GET'])
def get_test_dataset():
    """Get test dataset."""
    if TEST_DATA is None:
        return jsonify({'error': 'Test dataset not loaded'}), 500
    
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', type=int, default=0)
    
    data = TEST_DATA[offset:]
    if limit:
        data = data[:limit]
    
    return jsonify({
        'total': len(TEST_DATA),
        'returned': len(data),
        'offset': offset,
        'data': data
    })


@app.route('/api/dataset/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics."""
    if TRAIN_DATA is None or TEST_DATA is None:
        return jsonify({'error': 'Datasets not loaded'}), 500
    
    from collections import Counter
    
    train_entities = Counter()
    test_entities = Counter()
    
    for sample in TRAIN_DATA:
        for entity in sample.get('entities', []):
            train_entities[entity['label']] += 1
    
    for sample in TEST_DATA:
        for entity in sample.get('entities', []):
            test_entities[entity['label']] += 1
    
    return jsonify({
        'train': {
            'samples': len(TRAIN_DATA),
            'total_entities': sum(train_entities.values()),
            'entity_distribution': dict(train_entities)
        },
        'test': {
            'samples': len(TEST_DATA),
            'total_entities': sum(test_entities.values()),
            'entity_distribution': dict(test_entities)
        }
    })


@app.route('/api/dataset/sample', methods=['GET'])
def get_sample():
    """Get a random sample from dataset."""
    import random
    
    dataset_type = request.args.get('type', 'train')
    dataset = TRAIN_DATA if dataset_type == 'train' else TEST_DATA
    
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    if not dataset:
        return jsonify({'error': 'Dataset is empty'}), 404
    
    sample = random.choice(dataset)
    return jsonify(sample)


if __name__ == '__main__':
    print("Loading datasets...")
    load_datasets()
    print("\nStarting NER Dataset API Server...")
    print("API available at http://localhost:5000")
    print("API documentation at http://localhost:5000/")
    print("\nPress Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)

