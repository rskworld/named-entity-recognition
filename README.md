# Named Entity Recognition Dataset

<!--
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: NER dataset with labeled entities for information extraction and NLP applications
-->

## Overview

This dataset contains text documents with labeled named entities including persons, organizations, locations, dates, and other entity types. Perfect for named entity recognition, information extraction, and NLP model training.

## Features

- ✅ Labeled entities
- ✅ Multiple entity types
- ✅ BIO tagging format
- ✅ Training and test sets
- ✅ Ready for NER models

## Dataset Structure

```
named-entity-recognition/
├── README.md
├── index.html              # Landing page with header and footer
├── demo.html               # Interactive demo page
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── train.json
│   ├── test.json
│   └── train_bio.txt
├── scripts/
│   ├── load_dataset.py     # Basic dataset loading
│   ├── visualize_ner.py    # Entity visualization
│   ├── train_model.py      # Model training
│   ├── batch_process.py    # Advanced batch processing
│   ├── api_server.py       # Flask API server
│   ├── advanced_stats.py   # Comprehensive statistics
│   ├── export_data.py      # Multi-format export
│   └── evaluate_model.py   # Model evaluation
└── requirements.txt
```

## Entity Types

- **PERSON**: Names of people
- **ORG**: Organizations, companies, institutions
- **LOC**: Locations, cities, countries
- **DATE**: Dates and time expressions
- **MONEY**: Monetary values
- **PERCENT**: Percentages

## Data Formats

### CSV Format
The CSV files contain columns: `text`, `entity_type`, `start`, `end`, `entity_text`

### JSON Format
The JSON files contain structured data with sentences and their labeled entities.

### BIO Format
The BIO (Beginning-Inside-Outside) tagging format is used for sequence labeling tasks.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Loading the Dataset

```python
from scripts.load_dataset import load_ner_dataset

# Load training data
train_data = load_ner_dataset('dataset/train.json')
```

### Training a Model

```python
from scripts.train_model import train_ner_model

model = train_ner_model('dataset/train.json', 'dataset/test.json')
```

## Advanced Features

### 🚀 Batch Processing

Process large datasets efficiently with parallel processing:

```bash
python scripts/batch_process.py --input dataset/train.json --output processed_data.json --batch-size 10 --workers 4
```

### 🌐 REST API Server

Start a Flask API server for NER operations:

```bash
python scripts/api_server.py
```

API Endpoints:
- `GET /` - API information
- `POST /api/extract` - Extract entities from text
- `GET /api/dataset/train` - Get training dataset
- `GET /api/dataset/test` - Get test dataset
- `GET /api/dataset/stats` - Get dataset statistics
- `GET /api/dataset/sample` - Get random sample

### 📊 Advanced Statistics

Generate comprehensive statistics and analytics:

```bash
python scripts/advanced_stats.py --train dataset/train.json --test dataset/test.json --visualize
```

Features:
- Text analysis (length, word count distributions)
- Entity analysis (type distribution, coverage)
- Quality metrics (overlapping entities, validation)
- Visualization charts

### 📤 Export Functionality

Export dataset to multiple formats:

```bash
# Export to CSV
python scripts/export_data.py --input dataset/train.json --output train.csv --format csv

# Export to TSV
python scripts/export_data.py --input dataset/train.json --output train.tsv --format tsv

# Export to XML
python scripts/export_data.py --input dataset/train.json --output train.xml --format xml

# Export to BIO format
python scripts/export_data.py --input dataset/train.json --output train_bio.txt --format bio

# Export to CoNLL format
python scripts/export_data.py --input dataset/train.json --output train.conll --format conll

# Export to JSONL
python scripts/export_data.py --input dataset/train.json --output train.jsonl --format jsonl
```

### 🎯 Model Evaluation

Evaluate model predictions with detailed metrics:

```bash
python scripts/evaluate_model.py --true dataset/test.json --pred predictions.json --output evaluation_results.json
```

Metrics included:
- Precision, Recall, F1 Score
- Per-entity-type metrics
- Sample-level statistics
- True Positives, False Positives, False Negatives

### 📈 Visualization

Create visualizations of entities and statistics:

```python
from scripts.visualize_ner import create_visualization_report

data = load_ner_dataset('dataset/train.json')
create_visualization_report(data, 'ner_visualization.html')
```

## Technologies

- CSV
- JSON
- spaCy
- NLTK
- Transformers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This dataset is provided for educational and research purposes.

## Contact

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

---

*Created by RSK World - Free Programming Resources & Source Code*

