# Overview

This script is designed to generate explanations using a language model based on specified parameters and datasets. It allows users to customize various aspects of the explanation generation process through command-line arguments.


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

To evaluate the results type:
```bash
python src/evaluate.py
```

## Data Folder
To download the dataset please use this link: https://huggingface.co/datasets/Anon30241/model_kd_llm

