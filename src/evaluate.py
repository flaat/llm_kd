"""
Evaluation script for comparing performance of plain vs. finetuned models.

This module evaluates model responses for counterfactual explanation tasks on the Adult dataset.
It extracts JSON responses from model outputs, compares them to ground truth, and calculates
various performance metrics including:
- Average Feature Fidelity (Avg. FF) 
- Standard Deviation of Feature Fidelity (Std. FF)
- Perfect Feature Matches (PFM)
- Target Fidelity (TF)

Results are compiled into a LaTeX table format for easy inclusion in papers.
"""
import json
import re
import os
import glob
import statistics as stats
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Union, Tuple, Any, Optional


def load_json(file_path: str) -> Dict:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_and_parse_json(text: str, file: str) -> Optional[Dict]:
    """
    Extracts a JSON object from text and parses it into a dictionary.

    Args:
        text: The input string containing JSON-formatted content
        file: The filename (for debugging purposes)

    Returns:
        Parsed JSON as a dictionary if successful, else None
    """
    try:
        # Attempt to extract JSON block within triple backticks
        json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)

        if not json_match:
            # Attempt to extract JSON directly enclosed in curly brackets
            json_match = re.search(r"({.*})", text, re.DOTALL)

        if json_match:
            json_string = json_match.group(1).strip()
            return json.loads(json_string)
        else:
            print(f"⚠️ No JSON block found in the given text for {file}")
            return None

    except json.JSONDecodeError:
        print(f"⚠️ JSON parsing error in {file}")
        return None


def merge_dicts(dict_list: List[Dict]) -> Dict:
    """
    Merge a list of dictionaries into a structured dictionary with feature changes and reasonings.
    
    Args:
        dict_list: List of dictionaries to merge
        
    Returns:
        Dictionary with "feature_changes" and "reasonings" keys
    """
    result = {}
    reasonings = {}
    for d in dict_list:
        for key, value in d.items():
            if key == 'reasoning':
                continue  # Skip reasoning if at top-level
            result[key] = value
            if 'reasoning' in d:
                reasonings[key] = d['reasoning']
    return {"feature_changes": result, "reasonings": reasonings}


def process_model_responses(file_path: str) -> Tuple[Dict, Dict]:
    """
    Process model responses from a JSON file and calculate performance metrics.
    
    Args:
        file_path: Path to the file with model responses
        
    Returns:
        Tuple containing (metrics, answers)
    """
    data = load_json(file_path)    
    total_features_correct = 0
    total_target_correct = 0
    perfect_features_match = 0
    features_correct_list = []
    answers = {}
    
    for key, value in data.items():   
        values = extract_and_parse_json(value["generated_text"], file_path)
        if values is None:
            continue
            
        try:
            answers[key] = f'{values["reasoning"]} {values["explanation"]} {str(value["ground_truth"])}'
        except Exception:
            pass
            
        try:
            merged_dict = merge_dicts(values["feature_changes"])
            features_counter = 0
            changes = value.get('changes', [])
            
            if len(merged_dict["feature_changes"]) != len(changes["feature_changes"]) - 1:
                continue
                
            # Process each feature change
            for idx, element in enumerate(changes["feature_changes"]): 
                for variable, dictionary in element.items():
                    factual_ground_truth = dictionary["factual"]
                    counterfactual_ground_truth = dictionary["counterfactual"]
                    
                    if variable == "income":
                        try:  
                            check_factual_target = factual_ground_truth == int(values["target_variable_change"]["factual"])
                            check_counterfactual_target = counterfactual_ground_truth == int(values["target_variable_change"]["counterfactual"])
                            if check_factual_target and check_counterfactual_target:
                                total_target_correct += 1
                        except Exception:
                            pass
                    else:
                        try:
                            check_factual = factual_ground_truth == merged_dict["feature_changes"][variable]["factual"]
                            check_counterfactual = counterfactual_ground_truth == merged_dict["feature_changes"][variable]["counterfactual"]                    
                        
                            if check_factual and check_counterfactual:
                                features_counter += 1
                        except Exception:
                            pass
                    
            # Calculate feature fidelity metrics
            if features_counter == len(changes["feature_changes"]) - 1:
                perfect_features_match += 1
                
            feature_ratio = features_counter / (len(changes["feature_changes"]) - 1)
            total_features_correct += feature_ratio
            features_correct_list.append(feature_ratio)
        except Exception as e:
            print(f"Error processing entry {key}: {e}")
    
    try:
        stdv = round(stats.stdev(features_correct_list), 2) if features_correct_list else "n.d."
    except Exception:
        stdv = "n.d."
        
    metrics = {
        "features_correct": total_features_correct,
        "target_correct": total_target_correct,
        "perfect_matches": perfect_features_match,
        "std_deviation": stdv,
        "total_entries": len(data),
        "processed_entries": len(features_correct_list)
    }
    
    return metrics, answers


def generate_latex_table(overall_results: Dict) -> str:
    """
    Generate a LaTeX table from the results.
    
    Args:
        overall_results: Dictionary containing performance metrics for each model
        
    Returns:
        String containing LaTeX table code
    """
    output = "\\begin{table*}[htbp]\n \
    \\centering\n \
    \\begin{tabular}{l|p{2cm}|p{2cm}| p{2cm}|p{2cm}}\n \
        \\textbf{Model} & \\textbf{Avg. FF} $\\uparrow$ & \\textbf{Std. FF} $\\downarrow$ & \\textbf{Perfect FF} $\\uparrow$  & \\textbf{TF} $\\uparrow$  \\\\ \n\
        \\midrule \n \
        & \\footnotesize (Plain / Finetuned) &  \\footnotesize (Plain / Finetuned) &  \\footnotesize (Plain / Finetuned) & \\footnotesize (Plain / Finetuned) \\\\ \n\
        \\midrule \n"
        
    for key, value in overall_results.items():
        output += f"{key} & {value['Avg. FF']['plain']} / {value['Avg. FF']['finetuned']} & " \
                  f"{value['Std FF']['plain']} / {value['Std FF']['finetuned']} & " \
                  f"{value['PFM']['plain']} / {value['PFM']['finetuned']} & " \
                  f"{value['TF']['plain']} / {value['TF']['finetuned']} \\\\\n"
    
    output += "    \\end{tabular} \n " \
              "\\caption{Comparison of results for plain vs. finetuned models on the Adult dataset.}\n " \
              "\\label{tab:results-comparison}\n  " \
              "\\end{table*}"
              
    return output


def main():
    """Main function to evaluate model performance and generate result table."""
    # Define input directory
    results_dir = "data/results/evaluation_adult/"
    onlyfiles = [f for f in listdir(results_dir) if isfile(join(results_dir, f))]
    
    overall_results = {}
    all_answers = {}
    
    # Process each model file
    for file_name in sorted(onlyfiles):
        file_path = join(results_dir, file_name)
        metrics, answers = process_model_responses(file_path)
        all_answers.update(answers)
        
        # Extract model information
        model_name = file_name.split('_')[0] 
        model_type = "plain" if "Finetuned_False" in file_name else "finetuned"
        
        # Initialize model entry if not exists
        if model_name not in overall_results:
            overall_results[model_name] = {
                "Avg. FF": {"plain": None, "finetuned": None}, 
                "Std FF": {"plain": None, "finetuned": None}, 
                "TF": {"plain": None, "finetuned": None}, 
                "PFM": {"plain": None, "finetuned": None}
            }
        
        # Calculate and store metrics
        total_entries = metrics["total_entries"]
        processed_entries = metrics["processed_entries"]
        
        if processed_entries > 0:
            overall_results[model_name]["Avg. FF"][model_type] = round(metrics["features_correct"] / processed_entries, 3)
            overall_results[model_name]["Std FF"][model_type] = metrics["std_deviation"]
            overall_results[model_name]["PFM"][model_type] = round(metrics["perfect_matches"] / processed_entries, 3)
            overall_results[model_name]["TF"][model_type] = round(metrics["target_correct"] / processed_entries, 3)
    
    # Save answers dictionary to a JSON file
    with open('data/results/answers.json', 'w') as f:
        json.dump(all_answers, f, indent=4)
    
    # Generate and print LaTeX table
    latex_table = generate_latex_table(overall_results)
    print(latex_table)


if __name__ == '__main__':
    main()

