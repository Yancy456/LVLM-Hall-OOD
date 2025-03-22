# **LVLM-Hall-OOD**

This repository contains the source code for the paper.

## Getting Started

### Installation
####  Clone this repository to your local machine.


### Usage
We evaluate the LVLM-Hall-OOD method on four different datasets with six different models. For a specific dataset, you need to (i) prepare the dataset, (ii) run the model on the dataset to get answers and embeddings (iii) evaluate the performance of LVLM-Hall-OOD method. 

Don't worry. We are trying our best to make the process very simple for you.

#### 1. Download the dataset
After downloading and unzipping data, please modify dataset paths [here](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/blob/main/dataset/__init__.py).

- Task 1. Detect hallucinations in visual-answering dataset: VQA v2. Download the annotation file from [here] (https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip). Donwload coco2014 dataset from [Kaggle](https://www.kaggle.com/datasets/yashfinulhoque/coco-dataset-2014) or [offical website](https://visualqa.org/download.html). Modify the *annotation_path* and *data_folder*  to the annotation json file and the folder of coco2014 training set respectly.

#### 2. Run on different tasks

OK, life goes very easy at this stage. We provide you all configurations for running models on each task. Please find them [here](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/tree/main/scripts).

For example, run this command (modify it based on your gpu numbers)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/VizWiz/run_LLaVA_7B.sh
```

#### 3. Evaluate results

Please refer to those eval.ipynb files.


