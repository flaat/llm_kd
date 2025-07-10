MODEL_MAPPING = {
    "phi_4B": "microsoft/Phi-3.5-mini-instruct",  # 128k
    "phi_7B": "microsoft/Phi-3-small-128k-instruct",  # 128k
    "phi_14B": "microsoft/Phi-3-medium-128k-instruct",  # 128k
    "phi_14B_Q8_GGUF": "ssmits/Phi-3-medium-128k-instruct-Q8_0-GGUF",  # 128k
    "mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",  # 32k
    "smollm2_2B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # 8k
    "llama_8B": "meta-llama/Llama-3.1-8B-Instruct",  # 128k
    "llama_3B": "meta-llama/Llama-3.2-3B-Instruct",  # 128k
    "qwen_0.5B": "Qwen/Qwen2.5-0.5B-Instruct",  # 128k
    "qwen_1.5B": "Qwen/Qwen2.5-1.5B-Instruct",  # 128k
    "qwen_3B": "Qwen/Qwen2.5-3B-Instruct",  # 128k
    "qwen_7B": "Qwen/Qwen2.5-7B-Instruct",  # 128k
    "qwen_14B": "Qwen/Qwen2.5-14B-Instruct",  # 128k
    "qwen_32B": "Qwen/Qwen2.5-32B-Instruct",  # 128k
    "qwen_72B": "Qwen/Qwen2.5-72B-Instruct",  # 128k

    "qwen_0.5B_Q8_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",  # 128k
    "qwen_1.5B_Q8_GPTQ": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",  # 128k
    "qwen_3BQ_Q8_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",  # 128k
    "qwen_7BQ_Q8_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",  # 128k
    "qwen_14BQ_Q8_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",  # 128k
    "qwen_32BQ_Q8_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",  # 128k
    "qwen_72BQ_Q8_GPTQ": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",  # 128k
    
    "qwen_0.5B_Q4_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",  # 128k
    "qwen_1.5B_Q4_GPTQ": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",  # 128k
    "qwen_3BQ_Q4_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",  # 128k
    "qwen_7BQ_Q4_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  # 128k
    "qwen_14BQ_Q4_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",  # 128k
    "qwen_32BQ_Q4_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",  # 128k
    "qwen_72BQ_Q4_GPTQ": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",  # 128k

    "deepseek_r1_qwen_32B_Q4_AWQ1": "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    "deepseek_r1_qwen_32B_Q4_AWQ2": "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    "deepseek_r1_qwen_32B_Q4_GPTQ": "numen-tech/DeepSeek-R1-Distill-Qwen-32B-GPTQ-Int4",
    "deepseek_r1_qwen_32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_r1_qwen_3B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-3B",
    "deepseek_r1_qwen_7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    
    "unsloth_deepseek_r1_qwen_1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "unsloth_deepseek_r1_qwen_7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "unsloth_qwen_0.5B": "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth_qwen_3B": "unsloth/Qwen2.5-3B-Instruct"

}


prompt = """
A counterfactual explanation refers to a type of explanation in machine learning and artificial intelligence that describes how altering certain input features can change the output of a model. It answers 'what if' scenarios by identifying minimal changes necessary to achieve a different desired outcome. Counterfactual explanations provide insights into the decision-making process of complex models, enhancing transparency and interpretability.
For example, consider a credit scoring model that denies a loan application. A counterfactual explanation might be: 'If your annual income had been $50,000 instead of $45,000, your loan would have been approved.' This helps the applicant understand what specific change could lead to a different decision.

Your task is to generate a comprehensive, natural language counterfactual explanation of the classification change when transitioning from a factual example to its counterfactual counterpart.

Given the following inputs:
- Dataset Description: Background knowledge about the dataset, including feature definitions, their significance, and statistical distributions.
- Factual Example: A specific instance from the dataset that was classified under the original conditions.
- Counterfactual Example: A modified version of the factual example where certain features have been altered, resulting in a different classification.

The explanation should:
1. Identify Feature Changes: List and describe the features that differ between the factual and counterfactual examples. You should follow the structure outlined below.
2. Reasoning: Carry out a reasoning step that is functional to generating the final summary, in particular:
    - Analyze Contribution of Features: Assess the influence of each changed feature on the classification outcome, leveraging dataset knowledge to justify its impact.
    - Highlight Interactions: Discuss any interactions between features that may have played a role in shifting the classification outcome.
3. Summarize Key Factors: Conclude with a concise summary of the most influential features and their role in altering the prediction. The summary should be approximately 250 words. Avoid using bullet points, lists, or numerical outlines. Provide your responses in complete sentences and paragraphs, explaining concepts clearly and concisely in a continuous flow. The summary should be clear, coherent, and provide an intuitive understanding of how the model's decision was influenced by the observed feature modifications.

Your output should follow the following JSON structure:
{{
    "feature_changes": [
        {{"<FEATURE_1>": {{"factual": "<FACTUAL_VALUE_1>", "counterfactual": "<COUNTERFACTUAL_VALUE_1>"}}}},
        ...
        {{"<FEATURE_N>": {{"factual": "<FACTUAL_VALUE_N>", "counterfactual": "<COUNTERFACTUAL_VALUE_N>"}}}},
    ],
    "target_variable_change": {{"factual": "<FACTUAL_TARGET>", "counterfactual": "<COUNTERFACTUAL_TARGET>"}}, 
    "reasoning": "<YOUR_REASONING>",
    "explanation": "<YOUR_SUMMARY>"
}}
Please remeber to include also the target variable in the feature_changes list.
Please avoid any further explanation, clarification or other unnecessary outputs, just provide the JSON. Here is your input:
### Dataset Description ###
{dataset_description}

### Factual Example ###
{factual_example}

### Counterfactual Example ###
{counterfactual_example}
""".strip()

prompt_ref = """
A counterfactual explanation refers to a type of explanation in machine learning and artificial intelligence that describes how altering certain input features can change the output of a model. It answers 'what if' scenarios by identifying minimal changes necessary to achieve a different desired outcome. Counterfactual explanations provide insights into the decision-making process of complex models, enhancing transparency and interpretability.
For example, consider a credit scoring model that denies a loan application. A counterfactual explanation might be: 'If your annual income had been $50,000 instead of $45,000, your loan would have been approved.' This helps the applicant understand what specific change could lead to a different decision.

Your task is to generate a comprehensive, natural language counterfactual explanation of the classification change when transitioning from a factual example to its counterfactual counterpart.

Given the following inputs:
- Dataset Description: Background knowledge about the dataset, including feature definitions, their significance, and statistical distributions.
- Factual Example: A specific instance from the dataset that was classified under the original conditions.
- Counterfactual Example: A modified version of the factual example where certain features have been altered, resulting in a different classification.
- Draft Explanations: Three independent draft explanations of the same factual/counterfactual pair. These explanations may overlap, complement, or partially contradict each other.

The explanation should:
1. Identify Feature Changes: List and describe the features that differ between the factual and counterfactual examples. You should follow the structure outlined below.
2. Reasoning: Carry out a reasoning step that is functional to generating the final summary, in particular:
    - Analyze Contribution of Features: Assess the influence of each changed feature on the classification outcome, leveraging dataset knowledge to justify its impact.
    - Highlight Interactions: Discuss any interactions between features that may have played a role in shifting the classification outcome.
3. Integrate Draft Explanations: Carefully review the three draft explanations. Extract the core claims and evidence presented in each. In particular:
   - Where explanations conflict or differ, resolve these contradictions by prioritizing statements best supported by the dataset description and the identified feature changes.
   - Merge complementary insights to create a unified, logically consistent explanation, avoiding redundancy and ensuring the final narrative flows coherently.
4. Summarize Key Factors: Conclude with a concise summary of the most influential features and their role in altering the prediction. The summary should be approximately 250 words. Avoid using bullet points, lists, or numerical outlines. Provide your responses in complete sentences and paragraphs, explaining concepts clearly and concisely in a continuous flow. The summary should be clear, coherent, and provide an intuitive understanding of how the model's decision was influenced by the observed feature modifications.

Your output should follow the following JSON structure:
{{
    "feature_changes": [
        {{"<FEATURE_1>": {{"factual": "<FACTUAL_VALUE_1>", "counterfactual": "<COUNTERFACTUAL_VALUE_1>"}}}},
        ...
        {{"<FEATURE_N>": {{"factual": "<FACTUAL_VALUE_N>", "counterfactual": "<COUNTERFACTUAL_VALUE_N>"}}}},
    ],
    "target_variable_change": {{"factual": "<FACTUAL_TARGET>", "counterfactual": "<COUNTERFACTUAL_TARGET>"}}, 
    "reasoning": "<YOUR_REASONING>",
    "explanation": "<YOUR_SUMMARY>"
}}
Please remeber to include also the target variable in the feature_changes list.
Please avoid any further explanation, clarification or other unnecessary outputs, just provide the JSON. Here is your input:
### Dataset Description ###
{dataset_description}

### Factual Example ###
{factual_example}

### Counterfactual Example ###
{counterfactual_example}

### Draft Explanation 1 ###
{draft_explanation_1}

### Draft Explanation 2 ###
{draft_explanation_2}

### Draft Explanation 3 ###
{draft_explanation_3}
""".strip()


prompt_itc = """
A counterfactual explanation refers to a type of explanation in machine learning and artificial intelligence that describes how altering certain input features can change the output of a model. It answers 'what if' scenarios by identifying minimal changes necessary to achieve a different desired outcome. Counterfactual explanations provide insights into the decision-making process of complex models, enhancing transparency and interpretability.
For example, consider a credit scoring model that denies a loan application. A counterfactual explanation might be: 'If your annual income had been $50,000 instead of $45,000, your loan would have been approved.' This helps the applicant understand what specific change could lead to a different decision.

Your task is to generate a comprehensive, natural language counterfactual explanation of the classification change when transitioning from a factual example to its counterfactual counterpart.

Given the following inputs:
- Dataset Description: Background knowledge about the dataset, including feature definitions, their significance, and statistical distributions.
- Factual Example: A specific instance from the dataset that was classified under the original conditions.
- Counterfactual Example: A modified version of the factual example where certain features have been altered, resulting in a different classification.

The explanation should:
1. Identify Feature Changes: List and describe the features that differ between the factual and counterfactual examples. You should follow the structure outlined below.
2. Summarize Key Factors: Conclude with a concise summary of the most influential features and their role in altering the prediction. The summary should be approximately 250 words. Avoid using bullet points, lists, or numerical outlines. Provide your responses in complete sentences and paragraphs, explaining concepts clearly and concisely in a continuous flow. The summary should be clear, coherent, and provide an intuitive understanding of how the model's decision was influenced by the observed feature modifications.

Your output should follow the following JSON structure:
{{
    "feature_changes": [
        {{"<FEATURE_1>": {{"factual": "<FACTUAL_VALUE_1>", "counterfactual": "<COUNTERFACTUAL_VALUE_1>"}}}},
        ...
        {{"<FEATURE_N>": {{"factual": "<FACTUAL_VALUE_N>", "counterfactual": "<COUNTERFACTUAL_VALUE_N>"}}}},
    ],
    "explanation": "<YOUR_SUMMARY>"
}}

Please avoid any further explanation, clarification or other unnecessary outputs, just provide the JSON. Here is your input:
### Dataset Description ###
{dataset_description}

### Factual Example ###
{factual_example}

### Counterfactual Example ###
{counterfactual_example}
""".strip()
