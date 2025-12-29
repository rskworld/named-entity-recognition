# Release Notes - v1.0.0

<!--
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Release notes for Named Entity Recognition Dataset v1.0.0
-->

## 🎉 Initial Release - v1.0.0

**Release Date:** 2026  
**Author:** RSK World  
**Website:** https://rskworld.in

---

## 📦 What's Included

### Dataset
- ✅ **50 Training Samples** - Comprehensive training data with labeled entities
- ✅ **30 Test Samples** - Test dataset for model evaluation
- ✅ **6 Entity Types** - PERSON, ORG, LOC, DATE, MONEY, PERCENT
- ✅ **Multiple Formats** - CSV, JSON, and BIO tagging formats

### Features

#### 🌐 Web Interface
- **Interactive Landing Page** (`index.html`)
  - Animated hero section
  - Feature showcase
  - Statistics dashboard
  - Interactive charts
  - Complete dataset browser
  - Use cases section
  - Technologies showcase

- **Interactive Demo Page** (`demo.html`)
  - Real-time entity extraction
  - Text analysis interface
  - Entity highlighting
  - Statistics display
  - Sample text library
  - Complete dataset viewer

#### 🐍 Python Scripts
- **load_dataset.py** - Dataset loading utilities
- **visualize_ner.py** - Entity visualization tools
- **train_model.py** - Model training with spaCy
- **batch_process.py** - Advanced batch processing with parallel execution
- **api_server.py** - Flask REST API server
- **advanced_stats.py** - Comprehensive statistics and analytics
- **export_data.py** - Multi-format export (CSV, TSV, XML, BIO, CoNLL, JSONL)
- **evaluate_model.py** - Model evaluation and comparison tools

#### 📊 Advanced Features
- RESTful API with multiple endpoints
- Batch processing with progress tracking
- Advanced statistics and analytics
- Multiple export formats
- Model evaluation metrics
- Interactive visualizations
- Search and filter functionality

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python scripts/api_server.py

# View demo
# Open demo.html in your browser
```

## 📁 Project Structure

```
named-entity-recognition/
├── README.md
├── LICENSE
├── QUICKSTART.md
├── index.html              # Landing page
├── demo.html               # Interactive demo
├── requirements.txt
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── train.json
│   ├── test.json
│   └── train_bio.txt
└── scripts/
    ├── load_dataset.py
    ├── visualize_ner.py
    ├── train_model.py
    ├── batch_process.py
    ├── api_server.py
    ├── advanced_stats.py
    ├── export_data.py
    └── evaluate_model.py
```

## 🎯 Key Highlights

- **Complete Dataset:** 80 labeled samples ready for NER model training
- **Multiple Formats:** CSV, JSON, BIO, TSV, XML, CoNLL, JSONL
- **Interactive Demo:** Real-time entity extraction and visualization
- **REST API:** Full-featured Flask API server
- **Advanced Tools:** Batch processing, statistics, evaluation, export
- **Modern UI:** Animated, responsive, professional design
- **Comprehensive Documentation:** README, Quick Start, API docs

## 📝 Entity Types

- **PERSON** - Names of people
- **ORG** - Organizations and companies
- **LOC** - Locations and cities
- **DATE** - Dates and time expressions
- **MONEY** - Monetary values
- **PERCENT** - Percentage values

## 🔧 Technologies

- Python 3.x
- spaCy
- NLTK
- Transformers
- Flask
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

## 🙏 Acknowledgments

Created by RSK World - Free Programming Resources & Source Code

---

**Full Changelog:** This is the initial release of the Named Entity Recognition Dataset project.

