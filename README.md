# Overview

This project explores **two narrative generation pipelines** for Narrative Explainable AI:  
1. **Direct pipeline** – a Small Language Model (SLM) is fine-tuned to generate narratives directly from counterfactuals.  
2. **Multi-Narrative Refinement pipeline** – two SLMs are trained for two stages: first to produce multiple draft narratives (from the direct pipeline), then to refine them into a coherent explanation (refinement step).

![Pipelines](https://github.com/flaat/llm_kd/blob/06e40be90857a079eea6073bb6a4b22e7e13f2b2/data/pipelines_2.png)


## Requirements

- **Python 3.11.10**
- Install Required Python packages  using `pip install -r requirements.txt` 

## Usage

To fine tune the models you must run

```bash
bash finetune.sh
```

To generate the test results you must run
```bash
bash test.sh
```
or
```bash
bash test_refiner.sh
```

To evaluate the results type:
```bash
python src/evaluate.py
```
or
```bash
python src/evaluate_with_refiner.py
```

## Data Folder
The datasets were **synthetically generated with a Large Language Model (LLM)** and then distilled into SLMs through fine-tuning.
To download the datasets please use this link: https://huggingface.co/datasets/Anon30241/model_kd_llm

