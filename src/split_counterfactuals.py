#!/usr/bin/env python3
"""
Script to split counterfactual data into test and validation sets.
Takes the last 10 entries from each dataset in test_counterfactuals.json,
moves them to val_counterfactuals.json, and renames entries to start from 0.
"""

import json
from pathlib import Path


def split_and_rename_counterfactuals(input_file, test_output_file, val_output_file, val_size=10):
    """
    Split counterfactuals into test and validation sets, and rename entries to start from 0.
    
    Args:
        input_file: Path to the input JSON file
        test_output_file: Path to save the test set
        val_output_file: Path to save the validation set
        val_size: Number of last entries to move to validation set
    """
    # Load the input JSON
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize output dictionaries
    test_data = {}
    val_data = {}
    
    # Process each dataset
    for dataset_name, entries in data.items():
        print(f"Processing dataset: {dataset_name}")
        
        # Convert entries to a list to easily split
        entry_items = list(entries.items())
        
        # Split into test and validation
        test_entries = entry_items[:-val_size]
        val_entries = entry_items[-val_size:]
        
        # Convert back to dictionaries with renamed keys starting from 0
        test_data[dataset_name] = {str(idx): value for idx, (_, value) in enumerate(test_entries)}
        val_data[dataset_name] = {str(idx): value for idx, (_, value) in enumerate(val_entries)}
        
        print(f"  - Total entries: {len(entry_items)}")
        print(f"  - Test entries: {len(test_entries)} (renamed 0 to {len(test_entries)-1})")
        print(f"  - Validation entries: {len(val_entries)} (renamed 0 to {len(val_entries)-1})")
    
    # Save test data (modified original file)
    print(f"\nSaving test data to {test_output_file}...")
    with open(test_output_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    print("Test data saved!")
    
    # Save validation data
    print(f"Saving validation data to {val_output_file}...")
    with open(val_output_file, 'w') as f:
        json.dump(val_data, f, indent=4)
    print("Validation data saved!")
    
    print("\nSplit and rename complete!")


if __name__ == "__main__":
    # Define file paths
    base_dir = Path(__file__).parent
    input_file = base_dir / "src/explainer/test_counterfactuals.json"
    test_output_file = base_dir / "src/explainer/test_counterfactuals.json"
    val_output_file = base_dir / "src/explainer/val_counterfactuals.json"
    
    # Run the split and rename
    split_and_rename_counterfactuals(input_file, test_output_file, val_output_file)
