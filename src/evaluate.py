import json
import re
import os
import glob
import ast


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_and_parse_json(text, file):
    """
    Extracts a JSON object from a given text and parses it into a dictionary.

    Args:
        text (str): The input string containing JSON-formatted content.

    Returns:
        dict: Parsed JSON as a dictionary if successful, else None.
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
            print("⚠️ No JSON block found in the given text.")
            return None

    except json.JSONDecodeError as e:
        return None


def merge_dicts(dict_list):
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


def compute_feature_changes_from_prompt(prompt_text):
    """
    Try to extract factual and counterfactual examples from the prompt and
    build a feature_changes list and a target_variable_change dict in the
    same structure expected by the rest of the evaluator.

    Returns a tuple (feature_changes_list, target_variable_change_dict) or
    (None, None) if parsing failed.
    """
    if not prompt_text:
        return None, None
    try:
        # Try to locate labelled factual/counterfactual blocks first
        pattern = r"###\s*Factual Example\s*###\s*(\{.*?\})\s*###\s*Counterfactual Example\s*###\s*(\{.*?\})"
        m = re.search(pattern, prompt_text, re.DOTALL)
        if m:
            factual_str, counterfactual_str = m.group(1), m.group(2)
        else:
            # fallback: find the last two dict-like occurrences in the prompt
            dicts = re.findall(r"(\{.*?\})", prompt_text, re.DOTALL)
            if len(dicts) >= 2:
                factual_str, counterfactual_str = dicts[-2], dicts[-1]
            else:
                return None, None

        # Parse using ast.literal_eval to accept python-style dicts with single quotes
        factual = ast.literal_eval(factual_str)
        counterfactual = ast.literal_eval(counterfactual_str)

        # Determine target variable: prefer known names, else pick any key that changed
        target = None
        for t in ("income", "Survived", "survived", "target"):
            if t in factual:
                target = t
                break
        if target is None:
            # pick a key that differs between factual and counterfactual
            differing = [k for k in factual.keys() if k in counterfactual and factual[k] != counterfactual[k]]
            if len(differing) == 1:
                target = differing[0]
            elif len(differing) > 1:
                # if multiple differ, prefer a numeric/int-like one, else take the last
                for k in differing:
                    if isinstance(factual[k], int) or isinstance(counterfactual[k], int):
                        target = k
                        break
                if target is None:
                    target = differing[-1]

        # Build feature_changes list excluding the target variable and only
        # include features that actually change between factual and counterfactual.
        feature_changes = {}
        for k in factual.keys():
            if k == target:
                continue
            if k in counterfactual and factual[k] != counterfactual[k]:
                feature_changes[k] = {"factual": factual[k], "counterfactual": counterfactual[k]}

        target_var_change = None
        if target and target in factual and target in counterfactual:
            target_var_change = {"factual": factual[target], "counterfactual": counterfactual[target]}

        return feature_changes, target_var_change
    except Exception:
        return None, None

import statistics as stats
if __name__ == '__main__':  
    # Define file paths
    mypath = "data/results/evaluation_titanic_teacher/"
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] 
    output = "\\begin{table*}[htbp]\n \
    \\centering\n \
    \\begin{tabular}{l|p{2cm}|p{2cm}| p{2cm}|p{2cm}}\n \
        \\textbf{Model} & \\textbf{Avg. FF} $\\uparrow$ & \\textbf{Std. FF} $\\downarrow$ & \\textbf{Perfect FF} $\\uparrow$  & \\textbf{TF} $\\uparrow$  \\\\ \n\
        \\midrule \n \
        & \\footnotesize (Plain / Finetuned) &  \\footnotesize (Plain / Finetuned) &  \\footnotesize (Plain / Finetuned) & \\footnotesize (Plain / Finetuned) \\\\ \n\
        \\midrule \n"
    overall = {}
    answers = {}
    for file1_path in sorted(onlyfiles):
        # Load JSON data
        data = load_json(mypath+file1_path)    
        total_features_correct = 0
        total_target_correct = 0
        perfect_features_match = 0
        features_correct_list = []
        for key, value in data.items():   
            if int(key) > 199:
                break
            values = extract_and_parse_json(value["generated_text"], file1_path)
            if values is None:
                continue
            try:
                answers[key] = f'{values["reasoning"]} {values["explanation"]} {str(value["ground_truth"])}'
            except Exception as e:
                pass
            # Ensure we have feature_changes in the parsed values. If missing,
            # try to compute them from the original entry's prompt text.
            if not value.get("changes"):
                # Try several places for the prompt: the entry itself, the parsed values,
                # or fall back to the generated_text (which may contain the examples).
                value["changes"] = {}
                prompt_text = value.get("prompt") or values.get("prompt") or value.get("generated_text")
                fc, tv = compute_feature_changes_from_prompt(prompt_text)
                if fc:
                    value["changes"]["feature_changes"] = fc
                if tv:
                    value["changes"]["target_variable_change"] = tv

            merged_dict = merge_dicts(values["feature_changes"])
            features_counter = 0
            target_counter = 0
            # 'changes' is expected to be a dict with a 'feature_changes' list inside.
            changes = value.get('changes', {})
            
            if not isinstance(changes, dict) or 'feature_changes' not in changes:
                # can't compare without the expected 'changes' structure
                continue

            if len(merged_dict["feature_changes"]) != len(changes["feature_changes"]):
                continue
            for idx, (variable, element) in enumerate(changes["feature_changes"].items()): 
                    factual_ground_thruth = element["factual"]
                    counterfactual_ground_thruth = element["counterfactual"]
                    
                    if variable == "income" or variable == "Survived":
                        try:  
                            check_factual_target = factual_ground_thruth == int(values["target_variable_change"]["factual"])
                            check_counterfactual_target = counterfactual_ground_thruth == int(values["target_variable_change"]["counterfactual"])
                            if check_factual_target and check_counterfactual_target:
                                total_target_correct += 1
                        except Exception as e:
                            pass
                    else:
                        try:
                            check_factual = factual_ground_thruth == merged_dict["feature_changes"][variable]["factual"]
                            check_counterfactual = counterfactual_ground_thruth == merged_dict["feature_changes"][variable]["counterfactual"]                    
                        
                            if check_factual and check_counterfactual:
                                features_counter += 1
                        except Exception as e:
                            pass
            try:
                if int(values["target_variable_change"]["factual"]) == changes["target_variable_change"]["factual"] and int(values["target_variable_change"]["counterfactual"]) == changes["target_variable_change"]["counterfactual"]:
                    total_target_correct += 1  
            except Exception as e:
                pass                
            if features_counter == len(changes["feature_changes"]):
                perfect_features_match += 1
            total_features_correct += features_counter / (len(changes["feature_changes"]))
            features_correct_list.append(features_counter / (len(changes["feature_changes"])))
        model_name = file1_path.split('/')[-1].split('_')[0] 
        try:
            stdv = round(stats.stdev(features_correct_list), 2)
            
        except Exception as e:
            stdv = "n.d."
        
        model_type = "plain" if "Finetuned_False" in file1_path else "finetuned"
        if model_name not in overall:
            overall[model_name] =  {"Avg. FF": {"plain": None, "finetuned": None}, "Std FF": {"plain": None, "finetuned": None}, "TF": {"plain": None, "finetuned": None}, "PFM": {"plain": None, "finetuned": None}}
        overall[model_name]["Avg. FF"][model_type] = round(total_features_correct/200, 3)
        overall[model_name]["Std FF"][model_type] = stdv
        overall[model_name]["PFM"][model_type] = perfect_features_match/200
        overall[model_name]["TF"][model_type] = total_target_correct/200
    # Save answers dictionary to a JSON file
    with open('data/results/answers.json', 'w') as f:
        json.dump(answers, f, indent=4)
    for key, value in overall.items():
        output += f"{key} & {value['Avg. FF']['plain']} / {value['Avg. FF']['finetuned']} & {value['Std FF']['plain']} / {value['Std FF']['finetuned']} & {value['PFM']['plain']} / {value['PFM']['finetuned']} & {value['TF']['plain']} / {value['TF']['finetuned']} \\\\\n"
    output += "    \\end{tabular} \n \\caption{Comparison of results for plain vs. finetuned models on the titanic dataset.}\n \\label{tab:results-comparison}\n  \\end{table*}"
    print(output)
                
