# **HW3 - Dialogue-Based Photo Retrieval**

> Author: 312553024 江尚軒  
> Date: 2024/12/20

## Requirement

Please read [2024 Information Retrieval and Extraction HW3.pdf](https://github.com/AndyChiangSH/1131-generative-information-retrieval/blob/main/Homework/HW3/2024%20Information%20Retrieval%20and%20Extraction%20HW3.pdf) first to understand the requirements of this homework.

## Environment

1. Move into this folder
    
    ```bash
    cd Homework/HW3/
    ```
    
2. Create this conda environment
    
    ```bash
    conda env create -f environment.yml
    ```
    
3. Activate this conda environment
    
    ```bash
    conda activate 1131-generative-information-retrieval-HW3
    ```
    

## Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/2024-information-retrieval-extraction-homework-3/data)
2. Put the dataset in the `dataset/` folder

## Retriever

1. To retrieve the `photo_description` with `dialogue` by **TF-IDF**, please set the `CONFIG` and run this code
    
    ```bash
    python retriever/src/TF-IDF.py
    ```
    
2. To retrieve the `photo_description` with `dialogue` by **BM25**, please set the `CONFIG` and run this code
    
    ```bash
    python retriever/src/BM25.py
    ```
    
3. To retrieve the `photo_description` with `dialogue` by **BERTScore**, please set the `CONFIG` and run this code
    
    ```bash
    python retriever/src/BERTScore.py
    ```
    
4. The log file will be saved in the `retriever/log/` folder
5. The submission file will be saved in the `retriever/submission/` folder

## Image Captioning

1. To generate the `photo_description` from `photo_path` by **GIT**, please set the `CONFIG` and run this code
    
    ```bash
    python image_captioning/src/GIT.py
    ```
    
2. To generate the `photo_description` from `photo_path` by **BLIP**, please set the `CONFIG` and run this code
    
    ```bash
    python image_captioning/src/BLIP.py
    ```
    
3. To concatenate the `photo_description` from two datasets, please set the `CONFIG` and run this code
    
    ```bash
    python image_captioning/src/concat.py
    ```
    
4. The log file will be saved in the `image_captioning/log/` folder
5. The new dataset will be saved in the `image_captioning/dataset/` folder