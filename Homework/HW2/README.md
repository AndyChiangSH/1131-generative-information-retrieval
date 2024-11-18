# HW2 - Fact Checking

> Author: 312553024 江尚軒  
> Date: 2024/11/18

## Requirement

Please read [IRIE-HW2.pdf](https://github.com/AndyChiangSH/1131-generative-information-retrieval/blob/main/Homework/HW2/IRIE-HW2.pdf) first to understand the requirements of this homework.

## Environment

1. Move into this folder
    
    ```bash
    cd Homework/HW2/
    ```
    
2. Create this conda environment
    
    ```bash
    conda env create -f environment.yml
    ```
    
3. Activate this conda environment
    
    ```bash
    conda activate 1131-generative-information-retrieval-HW2
    ```
    

## Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/2024-generative-information-retrieval-hw-2/data)
2. Put the dataset in the `dataset/` folder

## Retriever

1. Modify arguments in the `retriever.py`
2. Run this code to retrieve the evidence sentences
    
    ```bash
    python src/retriever.py
    ```
    
3. The `train.json`, `valid.json`, and `test.json` with the evidence sentences will be saved in the `retriever/` folder

## Classifier

1. Modify configuration in the `classifier/config/` folder
2. Run this code to fine-tune the model with `train.json`
    
    ```bash
    python src/classifier.py
    ```
    
3. The fine-tuned model will be saved in the `classifier/model/` folder

## Evaluator

1. Modify configuration in the `classifier/config/` folder
2. Run this code to evaluate the model with `valid.json`
    
    ```bash
    python src/evaluator.py
    ```
    
3. The result and confusion matrix will be saved in the `evaluator/` folder

## Predictor

1. Modify configuration in the `classifier/config/` folder
2. Run this code to predict the label by fine-tuned model with `test.json`
    
    ```bash
    python src/predictor.py
    ```
    
3. The submission file will be saved in the `submission/` folder

## LLM

1. Modify arguments in the `llm.py`
2. Modify prompts in the `classifier/prompt/` folder
3. Run this code to predict the label by LLM with `test.json`
    
    ```bash
    python src/llm.py
    ```
    
4. The submission file will be saved in the `submission/` folder