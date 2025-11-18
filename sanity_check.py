#!/usr/bin/env python3
"""
Script to check for truncated generations in dataset JSON files.
Counts entries where generated_text doesn't end with '}', indicating token limit truncation.
"""

import json
import argparse
import os


def check_truncated_generations(dataset: str, model: str, type_name: str, delete: bool = False):
    """
    Check for truncated generations in the dataset JSON file.
    
    Args:
        dataset: Name of the dataset
        model: Name of the model
        type_name: Type of generation ('worker' or 'refiner')
        delete: If True, delete truncated entries from the dataset
    """
    # Construct filename
    filename = f"data/{dataset}_{type_name}_{model}.json"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"❌ Error: File '{filename}' not found.")
        return
    
    # Load JSON file
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse JSON file '{filename}': {e}")
        return
    except Exception as e:
        print(f"❌ Error: Failed to read file '{filename}': {e}")
        return
    
    # Count total entries and truncated entries
    total_entries = 0
    truncated_count = 0
    truncated_keys = []  # Store keys of truncated entries for deletion
    
    # Iterate through all entries
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
            
        if "generated_text" not in entry:
            print(f"⚠️  Warning: Entry '{key}' missing 'generated_text' field. Skipping.")
            continue
        
        generated_text = entry["generated_text"]
        
        # Check if it's a string
        if not isinstance(generated_text, str):
            print(f"⚠️  Warning: Entry '{key}' has non-string 'generated_text'. Skipping.")
            continue
        
        total_entries += 1
        
        # Check if text doesn't end with '}'
        if generated_text.strip() and not generated_text.rstrip().endswith('}'):
            truncated_count += 1
            truncated_keys.append(key)
    
    # Calculate percentage
    if total_entries == 0:
        print(f"❌ Error: No valid entries found in '{filename}'.")
        return
    
    percentage = (truncated_count / total_entries) * 100
    
    # Print results
    print(f"\n📊 Results for '{filename}':")
    print(f"   Total entries: {total_entries}")
    print(f"   Truncated entries (not ending with '}}'): {truncated_count}")
    print(f"   Percentage: {percentage:.2f}%")
    
    # Delete truncated entries if requested
    if delete and truncated_keys:
        print(f"\n🗑️  Deleting {len(truncated_keys)} truncated entries...")
        for key in truncated_keys:
            del data[key]
        
        # Save the updated data back to the file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"✅ Successfully deleted {len(truncated_keys)} entries and saved updated file.")
            print(f"   Remaining entries: {total_entries - len(truncated_keys)}")
        except Exception as e:
            print(f"❌ Error: Failed to save updated file: {e}")
    elif delete and not truncated_keys:
        print(f"\n✅ No truncated entries to delete.")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check for truncated generations in dataset JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python dataset_sanity_check.py --dataset adult --model qwen3_30B_A3B --type worker
  python dataset_sanity_check.py --dataset adult --model qwen3_30B_A3B --type refiner
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='adult',
        help='Name of the dataset (default: adult)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen3_30B_A3B',
        help='Name of the model (default: qwen3_30B_A3B)'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        default='worker',
        choices=['worker', 'refiner'],
        help='Type of generation: "worker" or "refiner" (default: worker)'
    )
    
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete truncated entries from the dataset'
    )
    
    args = parser.parse_args()
    
    check_truncated_generations(args.dataset, args.model, args.type, args.delete)


if __name__ == "__main__":
    main()

