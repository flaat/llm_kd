MODEL_MAPPING = {
    # microsoft phi
    "phi_4B": "microsoft/Phi-3.5-mini-instruct",  
    "phi_7B": "microsoft/Phi-3-small-128k-instruct",  
    "phi_14B": "microsoft/Phi-3-medium-128k-instruct",  
    "phi_14B_Q8_GGUF": "ssmits/Phi-3-medium-128k-instruct-Q8_0-GGUF",  
    # mistral
    "mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",  
    # smollm2
    "smollm2_2B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  
    # llama
    "llama_8B": "meta-llama/Llama-3.1-8B-Instruct",  
    "llama_3B": "meta-llama/Llama-3.2-3B-Instruct",  
    # qwen
    "qwen_0.5B": "Qwen/Qwen2.5-0.5B-Instruct",  
    "qwen_1.5B": "Qwen/Qwen2.5-1.5B-Instruct",  
    "qwen_3B": "Qwen/Qwen2.5-3B-Instruct",  
    "qwen_7B": "Qwen/Qwen2.5-7B-Instruct",  
    "qwen_14B": "Qwen/Qwen2.5-14B-Instruct",  
    "qwen_32B": "Qwen/Qwen2.5-32B-Instruct",  
    # qwen gptq-int8
    "qwen_0.5B_Q8_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",  
    "qwen_1.5B_Q8_GPTQ": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",  
    "qwen_3BQ_Q8_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",  
    "qwen_7BQ_Q8_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",  
    "qwen_14BQ_Q8_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",  
    "qwen_32BQ_Q8_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",  
    # qwen gptq-int4
    "qwen_0.5B_Q4_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",  
    "qwen_1.5B_Q4_GPTQ": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",  
    "qwen_3BQ_Q4_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",  
    "qwen_7BQ_Q4_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  
    "qwen_14BQ_Q4_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",  
    "qwen_32BQ_Q4_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",  
    # deepseek r1 qwen awq
    "deepseek_r1_qwen_32B_Q4_AWQ1": "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ",    #teacher model
    "deepseek_r1_qwen_32B_Q4_AWQ2": "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    "deepseek_r1_qwen_32B_Q4_GPTQ": "numen-tech/DeepSeek-R1-Distill-Qwen-32B-GPTQ-Int4", 
    "deepseek_r1_qwen_32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_r1_qwen_3B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-3B",
    "deepseek_r1_qwen_7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # unsloth deepseek r1 qwen
    "unsloth_deepseek_r1_qwen_1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",   #used in the paper
    "unsloth_deepseek_r1_qwen_7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",   #used in the paper
    "unsloth_qwen_0.5B": "unsloth/Qwen2.5-0.5B-Instruct",   #used in the paper
    "unsloth_qwen_3B": "unsloth/Qwen2.5-3B-Instruct",    #used in the paper

    # qwen3
    "qwen3_8B": "Qwen/Qwen3-8B-AWQ",
    "qwen3_30B_A3B": "QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ",
    "qwen3_32B": "Qwen/Qwen3-32B-AWQ"

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
Please remember to include also the target variable in the feature_changes list.

Final Instructions:
1. Do not include any JSON or list objects in your reasoning process.
2. Do not repeat or restate your reasoning, feature changes, or conclusions. Each point should appear only once in the final JSON.
3. Output only the final JSON object exactly in the required format, with no explanations, comments, or extra text before or after it.

Here is your input:
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
- Draft Explanations: Multiple independent draft explanations of the same factual/counterfactual pair. These explanations may overlap, complement, or partially contradict each other.

The explanation should:
1. Identify Feature Changes: List and describe the features that differ between the factual and counterfactual examples. You should follow the structure outlined below.
2. Reasoning: Carry out a reasoning step that is functional to generating the final summary, in particular:
    - Analyze Contribution of Features: Assess the influence of each changed feature on the classification outcome, leveraging dataset knowledge to justify its impact.
    - Highlight Interactions: Discuss any interactions between features that may have played a role in shifting the classification outcome.
3. Integrate Draft Explanations: Carefully review the draft explanations. Extract the core claims and evidence presented in each. In particular:
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
Please remember to include also the target variable in the feature_changes list.

Final Instructions:
- Output only the final JSON object exactly in the required format, with no explanations, comments, or extra text before or after it.

Here is your input:
### Dataset Description ###
{dataset_description}

### Factual Example ###
{factual_example}

### Counterfactual Examplele ###
{counterfactual_example}

{draft_narratives}
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
Please remember to include also the target variable in the feature_changes list.

Final Instructions:
- Output only the final JSON object exactly in the required format, with no explanations, comments, or extra text before or after it.

Here is your input:
### Dataset Description ###
{dataset_description}

### Factual Example ###
{factual_example}

### Counterfactual Example ###
{counterfactual_example}
""".strip()
