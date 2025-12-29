"""
Named Entity Recognition Dataset - Visualization Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Script to visualize named entities in text using spaCy and matplotlib
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import os


def load_dataset(file_path: str):
    """Load NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_entity_distribution(data, save_path='entity_distribution.png'):
    """
    Create a bar chart showing the distribution of entity types.
    
    Args:
        data: List of data samples with entities
        save_path: Path to save the visualization
    """
    entity_counts = Counter()
    for sample in data:
        for entity in sample.get('entities', []):
            entity_counts[entity['label']] += 1
    
    labels = list(entity_counts.keys())
    counts = list(entity_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
    plt.xlabel('Entity Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Named Entity Types', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def highlight_entities_html(text: str, entities: list) -> str:
    """
    Create HTML with highlighted entities.
    
    Args:
        text: Original text
        entities: List of entity dictionaries with start, end, label, text
        
    Returns:
        HTML string with highlighted entities
    """
    # Sort entities by start position (reverse to avoid index shifting)
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    colors = {
        'PERSON': '#FF6B6B',
        'ORG': '#4ECDC4',
        'LOC': '#45B7D1',
        'DATE': '#FFA07A',
        'MONEY': '#98D8C8',
        'PERCENT': '#F7DC6F'
    }
    
    html_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        label = entity['label']
        color = colors.get(label, '#CCCCCC')
        
        entity_text = html_text[start:end]
        highlighted = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;" title="{label}">{entity_text}</span>'
        html_text = html_text[:start] + highlighted + html_text[end:]
    
    return html_text


def create_visualization_report(data, output_file='ner_visualization.html'):
    """
    Create an HTML report with highlighted entities.
    
    Args:
        data: List of data samples with entities
        output_file: Path to save the HTML report
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NER Dataset Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .sample {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend {
                background: white;
                padding: 15px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend-item {
                display: inline-block;
                margin: 5px 10px;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            h1 {
                color: #333;
            }
            .text {
                line-height: 1.8;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <h1>Named Entity Recognition Dataset Visualization</h1>
        <div class="legend">
            <h3>Entity Types:</h3>
            <span class="legend-item" style="background-color: #FF6B6B;">PERSON</span>
            <span class="legend-item" style="background-color: #4ECDC4;">ORG</span>
            <span class="legend-item" style="background-color: #45B7D1;">LOC</span>
            <span class="legend-item" style="background-color: #FFA07A;">DATE</span>
            <span class="legend-item" style="background-color: #98D8C8;">MONEY</span>
            <span class="legend-item" style="background-color: #F7DC6F;">PERCENT</span>
        </div>
    """
    
    for i, sample in enumerate(data[:20], 1):  # Show first 20 samples
        highlighted_text = highlight_entities_html(sample['text'], sample['entities'])
        html_content += f"""
        <div class="sample">
            <h3>Sample {i}</h3>
            <div class="text">{highlighted_text}</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization report saved to {output_file}")


if __name__ == '__main__':
    train_path = '../dataset/train.json'
    
    if os.path.exists(train_path):
        print("Loading dataset for visualization...")
        data = load_dataset(train_path)
        
        print("Creating entity distribution chart...")
        visualize_entity_distribution(data)
        
        print("Creating HTML visualization report...")
        create_visualization_report(data)
        
        print("\nVisualization complete!")

