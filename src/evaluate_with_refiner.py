import json
import re
import os
import statistics as stats
from os import listdir
from os.path import isfile, join

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_and_parse_json(text, file):
    try:
        json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if not json_match:
            json_match = re.search(r"({.*})", text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1).strip()
            return json.loads(json_string)
        else:
            print("⚠️ No JSON block found in the given text.")
            return None
    except json.JSONDecodeError:
        return None

def merge_dicts(dict_list):
    result = {}
    reasonings = {}
    # Se dict_list è un dizionario, convertilo in una lista di dizionari
    if isinstance(dict_list, dict):
        dict_list = [dict_list]
    for d in dict_list:
        if not isinstance(d, dict):
            print(f"⚠️ Elemento non dizionario trovato: {d}")
            continue
        for key, value in d.items():
            if key == 'reasoning':
                continue
            result[key] = value
            if 'reasoning' in d:
                reasonings[key] = d['reasoning']
    return {"feature_changes": result, "reasonings": reasonings}

def parse_filename(file_name, is_refiner):
    if is_refiner:
        parts = file_name.replace('.json', '').split('_')
        worker_idx = parts.index('Worker') + 1
        refiner_idx = parts.index('Refiner') + 1
        worker_model = parts[worker_idx]
        refiner_model = parts[refiner_idx]
    else:
        worker_model = file_name.split('_')[0]
        refiner_model = "---"
    model_type = "plain" if "Finetuned_False" in file_name else "finetuned"
    return worker_model, refiner_model, model_type

if __name__ == '__main__':  
    results_dirs = [
        ("data/results/evaluation_adult/", False),
        ("data/results/evaluation_adult_refiner/", True)
    ]
    output = "\\begin{table*}[htbp]\n \
    \\centering\n \
    \\begin{tabular}{l|l|p{2cm}|p{2cm}| p{2cm}|p{2cm}}\n \
        \\textbf{Worker Model} & \\textbf{Refiner Model} & \\textbf{Avg. FF} $\\uparrow$ & \\textbf{Std. FF} $\\downarrow$ & \\textbf{Perfect FF} $\\uparrow$  & \\textbf{TF} $\\uparrow$  \\\\ \n\
        \\midrule \n \
        & & \\footnotesize (Plain / Finetuned) &  \\footnotesize (Plain / Finetuned) &  \\footnotesize (Plain / Finetuned) & \\footnotesize (Plain / Finetuned) \\\\ \n\
        \\midrule \n"
    overall = {}
    answers = {}
    for results_dir, is_refiner in results_dirs:
        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} not found, skipping...")
            continue
        onlyfiles = [f for f in listdir(results_dir) if isfile(join(results_dir, f))]
        for file1_path in sorted(onlyfiles):
            data = load_json(join(results_dir, file1_path))    
            total_features_correct = 0
            total_target_correct = 0
            perfect_features_match = 0
            features_correct_list = []
            for key, value in data.items():   
                values = extract_and_parse_json(value["generated_text"], file1_path)
                if values is None:
                    continue
                try:
                    answers[key] = f'{values["reasoning"]} {values["explanation"]} {str(value["ground_truth"])}'
                except Exception:
                    pass
                merged_dict = merge_dicts(values["feature_changes"])
                features_counter = 0
                changes = value.get('changes', [])
                if is_refiner and len(merged_dict["feature_changes"]) != len(changes["feature_changes"]):
                    print(f"⚠️ Mismatch in feature changes length for {file1_path}.")
                    continue
                elif not is_refiner and len(merged_dict["feature_changes"]) != len(changes["feature_changes"]) - 1:
                    print(f"⚠️ Mismatch in feature changes length for {file1_path}.")
                    continue
                for idx, element in enumerate(changes["feature_changes"]): 
                    for variable, dictionary in element.items():
                        factual_ground_truth = dictionary["factual"]
                        counterfactual_ground_truth = dictionary["counterfactual"]
                        if variable == "income" or variable == "Survived":
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
                if features_counter == len(changes["feature_changes"]) - 1:
                    perfect_features_match += 1
                total_features_correct += features_counter / (len(changes["feature_changes"]) - 1)
                features_correct_list.append(features_counter / (len(changes["feature_changes"]) - 1))
            worker_model, refiner_model, model_type = parse_filename(file1_path, is_refiner)
            model_key = (worker_model, refiner_model)
            try:
                stdv = round(stats.stdev(features_correct_list), 2)
            except Exception:
                stdv = "n.d."
            if model_key not in overall:
                overall[model_key] =  {"Avg. FF": {"plain": None, "finetuned": None}, "Std FF": {"plain": None, "finetuned": None}, "TF": {"plain": None, "finetuned": None}, "PFM": {"plain": None, "finetuned": None}}
            overall[model_key]["Avg. FF"][model_type] = round(total_features_correct/len(data), 3)
            overall[model_key]["Std FF"][model_type] = stdv
            overall[model_key]["PFM"][model_type] = perfect_features_match/len(data)
            overall[model_key]["TF"][model_type] = total_target_correct/len(data)
    with open('data/results/answers_with_refiner_adult.json', 'w') as f:
        json.dump(answers, f, indent=4)
    # Ordinamento per worker, poi refiner "---" prima
    sorted_keys = sorted(overall.keys(), key=lambda x: (x[0], x[1] == "---", x[1]))
    for worker_model, refiner_model in sorted_keys:
        value = overall[(worker_model, refiner_model)]
        output += f"{worker_model} & {refiner_model} & {value['Avg. FF']['plain']} / {value['Avg. FF']['finetuned']} & {value['Std FF']['plain']} / {value['Std FF']['finetuned']} & {value['PFM']['plain']} / {value['PFM']['finetuned']} & {value['TF']['plain']} / {value['TF']['finetuned']} \\\\\n"
    output += "    \\end{tabular} \n \\caption{Comparison of results for plain vs. finetuned models on the titanic dataset with and without refiner.}\n \\label{tab:results-comparison}\n  \\end{table*}"
    print(output)