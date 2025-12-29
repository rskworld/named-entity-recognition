# Quick Start Guide

<!--
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Quick start guide for Named Entity Recognition Dataset
-->

## 🚀 Getting Started

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. View the Demo

Open `demo.html` in your web browser to see the interactive NER demo.

Or visit `index.html` for the main landing page with full navigation.

### 3. Load the Dataset

```python
from scripts.load_dataset import load_ner_dataset

# Load training data
train_data = load_ner_dataset('dataset/train.json')
print(f"Loaded {len(train_data)} samples")
```

### 4. Start the API Server

```bash
cd scripts
python api_server.py
```

Then visit `http://localhost:5000` for API documentation.

### 5. Generate Statistics

```bash
python scripts/advanced_stats.py --train dataset/train.json --test dataset/test.json
```

### 6. Export Data

```bash
# Export to CSV
python scripts/export_data.py --input dataset/train.json --output output.csv --format csv
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [scripts/](scripts/) directory for advanced features
- Check out the [demo.html](demo.html) for interactive examples

## 🆘 Need Help?

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

