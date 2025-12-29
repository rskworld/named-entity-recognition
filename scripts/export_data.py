"""
Named Entity Recognition Dataset - Export Script
Project: Named Entity Recognition Dataset
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Export NER dataset to various formats (CSV, JSON, XML, TSV, etc.)
"""

import json
import csv
import os
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_to_csv(data: List[Dict[str, Any]], output_file: str):
    """Export dataset to CSV format."""
    rows = []
    for sample in data:
        text = sample.get('text', '')
        for entity in sample.get('entities', []):
            rows.append({
                'id': sample.get('id', ''),
                'text': text,
                'entity_text': entity.get('text', ''),
                'entity_type': entity.get('label', ''),
                'start': entity.get('start', 0),
                'end': entity.get('end', 0)
            })
    
    if rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'text', 'entity_text', 'entity_type', 'start', 'end'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} rows to {output_file}")
    else:
        print(f"No data to export to {output_file}")


def export_to_tsv(data: List[Dict[str, Any]], output_file: str):
    """Export dataset to TSV (Tab-Separated Values) format."""
    rows = []
    for sample in data:
        text = sample.get('text', '')
        for entity in sample.get('entities', []):
            rows.append({
                'id': sample.get('id', ''),
                'text': text,
                'entity_text': entity.get('text', ''),
                'entity_type': entity.get('label', ''),
                'start': entity.get('start', 0),
                'end': entity.get('end', 0)
            })
    
    if rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'text', 'entity_text', 'entity_type', 'start', 'end'], delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} rows to {output_file}")
    else:
        print(f"No data to export to {output_file}")


def export_to_xml(data: List[Dict[str, Any]], output_file: str):
    """Export dataset to XML format."""
    root = ET.Element('dataset')
    root.set('name', 'Named Entity Recognition Dataset')
    root.set('author', 'RSK World')
    root.set('website', 'https://rskworld.in')
    
    for sample in data:
        sample_elem = ET.SubElement(root, 'sample')
        sample_elem.set('id', str(sample.get('id', '')))
        
        text_elem = ET.SubElement(sample_elem, 'text')
        text_elem.text = sample.get('text', '')
        
        entities_elem = ET.SubElement(sample_elem, 'entities')
        for entity in sample.get('entities', []):
            entity_elem = ET.SubElement(entities_elem, 'entity')
            entity_elem.set('type', entity.get('label', ''))
            entity_elem.set('start', str(entity.get('start', 0)))
            entity_elem.set('end', str(entity.get('end', 0)))
            entity_elem.text = entity.get('text', '')
    
    # Pretty print
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    print(f"Exported {len(data)} samples to {output_file}")


def export_to_bio_format(data: List[Dict[str, Any]], output_file: str):
    """Export dataset to BIO tagging format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Named Entity Recognition Dataset - BIO Format\n")
        f.write("# Project: Named Entity Recognition Dataset\n")
        f.write("# Author: RSK World\n")
        f.write("# Website: https://rskworld.in\n")
        f.write("# Email: help@rskworld.in\n")
        f.write("# Phone: +91 93305 39277\n\n")
        
        for sample in data:
            text = sample.get('text', '')
            entities = sample.get('entities', [])
            
            # Create entity map
            entity_map = {}
            for entity in entities:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                label = entity.get('label', '')
                for pos in range(start, end):
                    if pos == start:
                        entity_map[pos] = f'B-{label}'
                    else:
                        entity_map[pos] = f'I-{label}'
            
            # Tokenize and assign tags
            words = text.split()
            char_pos = 0
            
            for word in words:
                word_start = char_pos
                word_end = char_pos + len(word)
                
                tag = entity_map.get(word_start, 'O')
                if tag == 'O':
                    # Check if word is inside an entity
                    for pos in range(word_start, word_end):
                        if pos in entity_map:
                            tag = entity_map[pos]
                            break
                
                f.write(f"{word}\t{tag}\n")
                char_pos = word_end + 1  # +1 for space
            
            f.write("\n")  # Blank line between sentences
    
    print(f"Exported {len(data)} samples to BIO format: {output_file}")


def export_to_conll_format(data: List[Dict[str, Any]], output_file: str):
    """Export dataset to CoNLL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in data:
            text = sample.get('text', '')
            entities = sample.get('entities', [])
            
            # Create entity map
            entity_map = {}
            for entity in entities:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                label = entity.get('label', '')
                for pos in range(start, end):
                    if pos == start:
                        entity_map[pos] = f'B-{label}'
                    else:
                        entity_map[pos] = f'I-{label}'
            
            # Tokenize and assign tags
            words = text.split()
            char_pos = 0
            
            for i, word in enumerate(words):
                word_start = char_pos
                word_end = char_pos + len(word)
                
                tag = entity_map.get(word_start, 'O')
                if tag == 'O':
                    for pos in range(word_start, word_end):
                        if pos in entity_map:
                            tag = entity_map[pos]
                            break
                
                # CoNLL format: word POS-tag chunk-tag NER-tag
                f.write(f"{i+1}\t{word}\t_\t_\t{tag}\n")
                char_pos = word_end + 1
            
            f.write("\n")
    
    print(f"Exported {len(data)} samples to CoNLL format: {output_file}")


def export_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    """Export dataset to JSONL (JSON Lines) format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Exported {len(data)} samples to JSONL format: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export NER dataset to various formats')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--format', type=str, choices=['csv', 'tsv', 'xml', 'bio', 'conll', 'jsonl'], 
                       required=True, help='Export format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    print(f"Loading dataset from {args.input}...")
    data = load_dataset(args.input)
    print(f"Loaded {len(data)} samples")
    
    print(f"\nExporting to {args.format.upper()} format...")
    
    if args.format == 'csv':
        export_to_csv(data, args.output)
    elif args.format == 'tsv':
        export_to_tsv(data, args.output)
    elif args.format == 'xml':
        export_to_xml(data, args.output)
    elif args.format == 'bio':
        export_to_bio_format(data, args.output)
    elif args.format == 'conll':
        export_to_conll_format(data, args.output)
    elif args.format == 'jsonl':
        export_to_jsonl(data, args.output)
    
    print(f"\nExport complete! Output saved to {args.output}")

