#!/usr/bin/env python3
"""
Sanity-check dataset JSON files for truncated generations and semantic correctness.

Validates that:
- generated_text outputs are complete (end with '}')
- feature_changes match the actual differences between factual and counterfactual samples
- target_variable_change aligns with the ground truth
- features_importance_ranking only references features present in feature_changes
"""

import json
import argparse
import os
import re
import ast
import copy


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
FEATURE_JSON_PATTERN = re.compile(r"(\{\s*\"feature_changes\".*\})", re.DOTALL)


def extract_examples_from_prompt(prompt_text: str):
    if not prompt_text:
        return None, None
    pattern = r"###\s*Factual Example\s*###\s*(\{.*?\})\s*###\s*Counterfactual Example\s*###\s*(\{.*?\})"
    match = re.search(pattern, prompt_text, re.DOTALL)
    if match:
        factual_str, counterfactual_str = match.group(1), match.group(2)
    else:
        dicts = re.findall(r"(\{.*?\})", prompt_text, re.DOTALL)
        if len(dicts) >= 2:
            factual_str, counterfactual_str = dicts[-2], dicts[-1]
        else:
            return None, None
    try:
        factual = ast.literal_eval(factual_str)
        counterfactual = ast.literal_eval(counterfactual_str)
        return factual, counterfactual
    except Exception:
        return None, None


def extract_ground_truth_changes(prompt_text: str):
    factual, counterfactual = extract_examples_from_prompt(prompt_text)
    if not factual or not counterfactual:
        return None
    diffs = {}
    shared_keys = set(factual.keys()).intersection(counterfactual.keys())
    for key in shared_keys:
        if factual[key] != counterfactual[key]:
            diffs[key] = {"factual": factual[key], "counterfactual": counterfactual[key]}
    return diffs


def extract_last_json_object(text: str):
    if not text:
        return None
    code_matches = list(JSON_BLOCK_PATTERN.finditer(text))
    if code_matches:
        match = code_matches[-1]
        candidate = match.group(1)
        try:
            return {
                "data": json.loads(candidate),
                "start": match.start(1),
                "end": match.end(1),
                "type": "codeblock",
                "block_start": match.start(),
                "block_end": match.end(),
            }
        except json.JSONDecodeError:
            pass
    feature_matches = list(FEATURE_JSON_PATTERN.finditer(text))
    if feature_matches:
        match = feature_matches[-1]
        candidate = match.group(1)
        try:
            return {
                "data": json.loads(candidate),
                "start": match.start(1),
                "end": match.end(1),
                "type": "inline",
            }
        except json.JSONDecodeError:
            pass
    idx = len(text)
    while idx != -1:
        idx = text.rfind("{", 0, idx)
        if idx == -1:
            break
        candidate = text[idx:]
        try:
            return {
                "data": json.loads(candidate),
                "start": idx,
                "end": len(text),
                "type": "inline",
            }
        except json.JSONDecodeError:
            idx -= 1
            continue
    return None


def flatten_feature_changes(feature_changes):
    flattened = {}
    if not isinstance(feature_changes, list):
        return flattened
    for change in feature_changes:
        if not isinstance(change, dict):
            continue
        for feature, values in change.items():
            if isinstance(values, dict):
                flattened[feature] = values
    return flattened


def normalize_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    # Try to convert string to float
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    return value


def values_match(expected, reported):
    if not isinstance(expected, dict) or not isinstance(reported, dict):
        return False
    return (
        normalize_value(expected.get("factual")) == normalize_value(reported.get("factual"))
        and normalize_value(expected.get("counterfactual")) == normalize_value(reported.get("counterfactual"))
    )


def find_matching_feature(expected_diffs, target_change):
    if not isinstance(target_change, dict):
        return None
    for feature, values in expected_diffs.items():
        if values_match(values, target_change):
            return feature
    return None


def compare_feature_changes(expected, reported):
    missing = []
    value_mismatches = []
    for feature, expected_values in expected.items():
        if feature not in reported:
            missing.append(feature)
        elif not values_match(expected_values, reported[feature]):
            value_mismatches.append(feature)
    extra = [feature for feature in reported.keys() if feature not in expected]
    return missing, extra, value_mismatches


def normalize_ranking_values(ranking: dict):
    if not isinstance(ranking, dict):
        return ranking
    ordered_unique = []
    for value in ranking.values():
        if value not in ordered_unique:
            ordered_unique.append(value)
    mapping = {value: idx + 1 for idx, value in enumerate(ordered_unique)}
    for feature, value in ranking.items():
        ranking[feature] = mapping[value]
    return ranking


def is_valid_ranking_sequence(ranking: dict):
    """
    Check that ranking values are numbered as 1..K with no gaps.
    Repetitions are allowed (e.g., [1, 1, 2, 3, 3] is valid).
    Also accepts string representations of integers (e.g., "1", "2").
    """
    if not isinstance(ranking, dict) or not ranking:
        return False, "Missing or empty ranking"

    values = list(ranking.values())
    
    # Convert string values to integers
    converted_values = []
    for v in values:
        if isinstance(v, int):
            converted_values.append(v)
        elif isinstance(v, str):
            try:
                converted_values.append(int(v))
            except ValueError:
                return False, f"Non-integer ranking value: {v}"
        else:
            return False, f"Invalid ranking value type: {type(v)}"

    unique_vals = sorted(set(converted_values))
    if unique_vals[0] != 1:
        return False, "Ranking must start at 1"

    if unique_vals[-1] != len(unique_vals):
        # There is a gap (e.g., 1, 3, 4 -> unique=[1,3,4], max=4, len=3)
        return False, f"Ranking must be consecutive integers 1..K (found {unique_vals})"

    return True, None


def prune_feature_changes(feature_changes_list, extras):
    if not isinstance(feature_changes_list, list):
        return feature_changes_list
    # Use case-insensitive comparison
    extras_set_lower = {extra.lower() for extra in extras}
    pruned = []
    for change in feature_changes_list:
        if not isinstance(change, dict):
            continue
        feature_name = next(iter(change.keys()), None)
        if feature_name and feature_name.lower() in extras_set_lower:
            continue
        pruned.append(change)
    return pruned


def replace_json_segment(entry, json_info, parsed_json):
    new_json_str = json.dumps(parsed_json, ensure_ascii=False, indent=4)
    text = entry.get("generated_text", "")
    if json_info["type"] == "codeblock":
        new_block = f"```json\n{new_json_str}\n```"
        entry["generated_text"] = (
            text[: json_info["block_start"]] + new_block + text[json_info["block_end"] :]
        )
    else:
        entry["generated_text"] = text[: json_info["start"]] + new_json_str + text[json_info["end"] :]


def clean_extraneous_features(entry, json_info, parsed_json, extras):
    if not extras:
        return False
    feature_changes_list = parsed_json.get("feature_changes")
    new_feature_changes = prune_feature_changes(feature_changes_list, extras)
    if feature_changes_list == new_feature_changes:
        return False
    parsed_json["feature_changes"] = new_feature_changes
    
    # After pruning feature_changes, also check ranking for any extras
    ranking = parsed_json.get("features_importance_ranking")
    if isinstance(ranking, dict):
        # Get the new set of feature names after pruning
        new_feature_names = set()
        for change in new_feature_changes:
            if isinstance(change, dict):
                new_feature_names.update(change.keys())
        
        # Create case-insensitive mapping
        new_feature_names_lower = {name.lower() for name in new_feature_names}
        
        # Remove from ranking any features not in the pruned feature_changes (case-insensitive)
        features_to_remove = [f for f in ranking.keys() if f.lower() not in new_feature_names_lower]
        
        for feature in features_to_remove:
            ranking.pop(feature, None)
        
        if not ranking:
            # If ranking is now empty, this entry should be removed
            return False
            
        parsed_json["features_importance_ranking"] = normalize_ranking_values(ranking)
    replace_json_segment(entry, json_info, parsed_json)
    return True


def check_truncated_generations(dataset: str, model: str, type_name: str, clean: bool = False, check_cleaned: bool = False):
    """
    Check for truncated generations in the dataset JSON file.
    
    Args:
        dataset: Name of the dataset
        model: Name of the model
        type_name: Type of generation ('worker' or 'refiner')
        clean: If True, attempt to clean the dataset (fix removable issues or drop bad samples)
    """
    # Construct filename
    if check_cleaned:
        filename = f"data/{dataset}_{type_name}_{model}_cleaned.json"
    else:   
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
    ground_truth_fail = []
    json_parse_fail = []
    feature_mismatch_details = []
    importance_mismatch_details = []
    feature_issue_keys = set()
    ranking_issue_keys = set()
    keys_to_remove = set()
    cleaned_keys = set()
    cleaned_data = copy.deepcopy(data) if clean else None
    
    prompt_tag_issue_count = 0
    
    # Iterate through all entries
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        
        current_entry = cleaned_data[key] if clean else entry
        
        # Check for prompt tags
        prompt_text = current_entry.get("prompt", "")
        if isinstance(prompt_text, str):
            start_tag = "<|im_start|>user\n"
            end_tag = "<|im_end|>\n<|im_start|>assistant\n"
            has_start = prompt_text.startswith(start_tag)
            has_end = prompt_text.endswith(end_tag)
            
            if has_start or has_end:
                prompt_tag_issue_count += 1
                if clean:
                    if has_start:
                        prompt_text = prompt_text[len(start_tag):]
                    if has_end:
                        prompt_text = prompt_text[:-len(end_tag)]
                    current_entry["prompt"] = prompt_text
                    cleaned_keys.add(key)

        if "generated_text" not in current_entry:
            print(f"⚠️  Warning: Entry '{key}' missing 'generated_text' field. Skipping.")
            if clean:
                keys_to_remove.add(key)
            continue
        
        generated_text = current_entry["generated_text"]
        
        # Check if it's a string
        if not isinstance(generated_text, str):
            print(f"⚠️  Warning: Entry '{key}' has non-string 'generated_text'. Skipping.")
            if clean:
                keys_to_remove.add(key)
            continue
        
        total_entries += 1
        
        # Check if text doesn't end with '}'
        if generated_text.strip() and not generated_text.rstrip().endswith('}'):
            truncated_count += 1
            truncated_keys.append(key)
            if clean:
                keys_to_remove.add(key)
            continue

        ground_truth = extract_ground_truth_changes(current_entry.get("prompt", ""))
        if ground_truth is None:
            ground_truth_fail.append(key)
            if clean:
                keys_to_remove.add(key)
            continue

        json_info = extract_last_json_object(generated_text)
        if json_info is None:
            json_parse_fail.append(key)
            if clean:
                keys_to_remove.add(key)
            continue

        parsed_json = json_info["data"]
        feature_changes_list = parsed_json.get("feature_changes")
        feature_changes = flatten_feature_changes(feature_changes_list)
        if not feature_changes:
            feature_mismatch_details.append(
                {"key": key, "reason": "Missing or invalid feature_changes block"}
            )
            feature_issue_keys.add(key)
            if clean:
                keys_to_remove.add(key)
            continue

        expected_changes = ground_truth
        missing, extra, value_mismatches = compare_feature_changes(expected_changes, feature_changes)
        target_change = parsed_json.get("target_variable_change")
        matching_feature = find_matching_feature(expected_changes, target_change)
        target_issue = matching_feature is None or matching_feature not in feature_changes

        extras_only_issue = bool(extra) and not missing and not value_mismatches and not target_issue
        if extras_only_issue and clean:
            cleaned = clean_extraneous_features(current_entry, json_info, parsed_json, extra)
            if cleaned:
                cleaned_keys.add(key)
                continue

        if missing or value_mismatches or (extra and not extras_only_issue):
            feature_mismatch_details.append(
                {
                    "key": key,
                    "missing": missing,
                    "extra": extra,
                    "value_mismatches": value_mismatches,
                }
            )
            feature_issue_keys.add(key)
            if clean:
                keys_to_remove.add(key)
            continue

        if target_issue:
            feature_mismatch_details.append(
                {"key": key, "reason": "target_variable_change does not match ground truth"}
            )
            feature_issue_keys.add(key)
            if clean:
                keys_to_remove.add(key)
            continue

        ranking = parsed_json.get("features_importance_ranking")
        if not isinstance(ranking, dict):
            importance_mismatch_details.append(
                {"key": key, "reason": "Missing or invalid features_importance_ranking"}
            )
            ranking_issue_keys.add(key)
            if clean:
                keys_to_remove.add(key)
            continue

        # Create case-insensitive mapping for feature names
        feature_change_names_lower = {name.lower(): name for name in feature_changes.keys()}
        
        # Find extras in ranking (case-insensitive)
        extras_in_ranking = []
        for rank_feature in ranking.keys():
            if rank_feature.lower() not in feature_change_names_lower:
                extras_in_ranking.append(rank_feature)
        
        if extras_in_ranking:
            if clean:
                changed = False
                for feature in extras_in_ranking:
                    if feature in ranking:
                        del ranking[feature]
                        changed = True
                if changed:
                    # Check if ranking is now empty after removal
                    if not ranking:
                        keys_to_remove.add(key)
                        continue
                    parsed_json["features_importance_ranking"] = normalize_ranking_values(ranking)
                    replace_json_segment(current_entry, json_info, parsed_json)
                    cleaned_keys.add(key)
                    continue
            importance_mismatch_details.append(
                {"key": key, "extra_features": sorted(extras_in_ranking)}
            )
            ranking_issue_keys.add(key)
            if clean:
                keys_to_remove.add(key)
            continue

        # Check that ranking values are correctly numbered 1..K (with possible repetitions)
        valid_ranking, reason = is_valid_ranking_sequence(ranking)
        if not valid_ranking:
            if clean:
                # In clean mode, normalize the ranking to fix numbering
                parsed_json["features_importance_ranking"] = normalize_ranking_values(ranking)
                replace_json_segment(current_entry, json_info, parsed_json)
                cleaned_keys.add(key)
                continue

            importance_mismatch_details.append(
                {"key": key, "reason": f"Invalid ranking sequence: {reason}"}
            )
            ranking_issue_keys.add(key)
            continue

    # Calculate percentage
    if total_entries == 0:
        print(f"❌ Error: No valid entries found in '{filename}'.")
        return
    
    truncated_percentage = (truncated_count / total_entries) * 100
    
    # Calculate detailed feature mismatch counts
    missing_only_count = 0
    extra_only_count = 0
    value_mismatch_only_count = 0
    mixed_mismatch_count = 0
    target_only_count = 0
    
    for detail in feature_mismatch_details:
        if detail.get('reason'):
            # This is a target_variable_change or feature_changes block issue
            if 'target_variable_change' in detail.get('reason', ''):
                target_only_count += 1
        else:
            missing = detail.get('missing', [])
            extra = detail.get('extra', [])
            value_mismatches = detail.get('value_mismatches', [])
            
            issue_count = sum([bool(missing), bool(extra), bool(value_mismatches)])
            
            if issue_count > 1:
                mixed_mismatch_count += 1
            elif missing:
                missing_only_count += 1
            elif extra:
                extra_only_count += 1
            elif value_mismatches:
                value_mismatch_only_count += 1
    
    # Print results
    print(f"\n📊 Results for '{filename}':")
    print(f"   Total entries: {total_entries}")
    print(f"   Truncated entries (not ending with '}}'): {truncated_count} ({truncated_percentage:.2f}%)")
    print(f"   Ground-truth parsing failures: {len(ground_truth_fail)}")
    print(f"   JSON parsing failures: {len(json_parse_fail)}")
    print(f"   Feature change mismatches: {len(feature_issue_keys)}")
    print(f"      - Missing features only: {missing_only_count}")
    print(f"      - Extra features only: {extra_only_count}")
    print(f"      - Value mismatches only: {value_mismatch_only_count}")
    print(f"      - Mixed issues: {mixed_mismatch_count}")
    print(f"      - Target variable mismatch: {target_only_count}")
    print(f"   Ranking mismatches: {len(ranking_issue_keys)}")
    print(f"   Prompt tag issues: {prompt_tag_issue_count}")
    if clean:
        print(f"   Entries cleaned (features/ranking/prompt): {len(cleaned_keys)}")
        print(f"   Entries removed: {len(keys_to_remove)}")
    
    if ground_truth_fail:
        print("   ⚠️  Unable to extract factual/counterfactual examples for keys:", ", ".join(ground_truth_fail[:5]), "..." if len(ground_truth_fail) > 5 else "")
    if json_parse_fail:
        print("   ⚠️  Failed to parse generated JSON for keys:", ", ".join(json_parse_fail[:5]), "..." if len(json_parse_fail) > 5 else "")
    if feature_mismatch_details:
        print("   ℹ️  Sample feature mismatch details:")
        for detail in feature_mismatch_details[:3]:
            print(f"      - Key {detail['key']}: {detail.get('reason') or ''} "
                  f"Missing={detail.get('missing')} Extra={detail.get('extra')} ValueMismatch={detail.get('value_mismatches')}")
        if len(feature_mismatch_details) > 3:
            print(f"      ... ({len(feature_mismatch_details) - 3} more)")
    if importance_mismatch_details:
        print("   ℹ️  Sample ranking mismatch details:")
        for detail in importance_mismatch_details[:3]:
            print(f"      - Key {detail['key']}: {detail.get('reason') or ''} ExtraFeatures={detail.get('extra_features')}")
        if len(importance_mismatch_details) > 3:
            print(f"      ... ({len(importance_mismatch_details) - 3} more)")
    
    if clean:
        for key in keys_to_remove:
            if key in cleaned_data:
                del cleaned_data[key]
        cleaned_filename = filename.replace('.json', '_cleaned.json')
        try:
            with open(cleaned_filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=4)
            print(f"\n🧼 Saved cleaned dataset to '{cleaned_filename}'.")
            print(f"   Entries cleaned: {len(cleaned_keys)}")
            print(f"   Entries removed: {len(keys_to_remove)}")
        except Exception as e:
            print(f"\n❌ Error: Failed to save cleaned file: {e}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run sanity checks on dataset JSON files (truncation, feature correctness, rankings)",
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
        '--clean',
        action='store_true',
        help='Create a cleaned copy: fix removable issues and drop irreparable samples'
    )

    parser.add_argument(
        '--check_cleaned',
        action='store_true',
        help='Create a cleaned copy: fix removable issues and drop irreparable samples'
    )
    
    args = parser.parse_args()
    
    check_truncated_generations(args.dataset, args.model, args.type, args.clean, args.check_cleaned)


if __name__ == "__main__":
    main()

