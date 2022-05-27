## Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT
> **Authors**: Zaiqiao Meng, Fangyu Liu, Thomas Hikaru Clark, Ehsan Shareghi, Nigel Collier.
> 
Code of our paper [Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT](https://arxiv.org/abs/2109.04810)[EMNLP2021]

## News:
[26 August 2022] - Our paper has been accepted to appear at the EMNLP 2021 as a short paper.

------
## Introduction
Infusing factual knowledge into pre-trained models is fundamental for many knowledge-intensive tasks. In this paper, we proposed Mixture-of-Partitions (MoP), an infusion approach that can handle a very large knowledge graph (KG) by partitioning it into smaller sub-graphs and infusing their specific knowledge into various BERT models using lightweight adapters. To leverage the overall factual knowledge for a target task, these sub-graph adapters are further fine-tuned along with the underlying BERT through a mixture layer. We evaluate our MoP with three biomedical BERTs (SciBERT, BioBERT, PubmedBERT) on six downstream tasks (inc. NLI, QA, Classification), and the results show that our MoP consistently enhances the underlying BERTs in task performance, and achieves new SOTA performances on five evaluated datasets.

 ![front-page-graph](/framework_double_col.jpg)


------

## File structure

- `data_dir`: downstream task dataset used in the experiments.
- `kg_dir`: folder to save the knowledge graphs as well as the partitioned files.
- `model_dir`: folder to save pre-trained models.
- `src`: source code.
  - `adapter-transformers`: adapter-transformers v1.1.1 forked from [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers), it has been modified for using different mixture approaches.
  - `evaluate_tasks`: codes for the downstream tasks.
  -  `knowledge_infusion`: knowledge infusion main codes.

kg_dir and model_dir can be downloaded at this [link](https://www.dropbox.com/s/s8zb8o5agpgx1e9/data_model.zip?dl=0).
## Installation
The code is tested with python 3.8.5, torch 1.7.0 and huggingface transformers 3.5.0. Please view requirements.txt for more details.
Our models use a modified [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers). To use this package, please go to the `./src/adapter-transformers` folder of this project, and run `pip install .` to install the `adapter-transformers` package

## Datasets
- The BioAsq7b, PubMedQA, HoC datasets can be downloaded from [BLURB](https://microsoft.github.io/BLURB/submit.html)
- The MedQA dataset can be downloaded from: https://github.com/jind11/MedQA
- The BioAsq8b datasets can be downloaded from: http://bioasq.org/

## Train knowledge fusion and downstream tasks

### Train Knowledge Infusion
To train knowledge infusion, you can run the following command in the `src/knowledge_infusion/entity_prediction` folder.
<details>
<summary>Click to expand!</summary>
  
```shell
MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TOKENIZER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
INPUT_DIR="kg_dir"
OUTPUT_DIR="model_dir"
DATASET_NAME="S20Rel"
ADAPTER_NAMES="entity_predict"
PARTITION=20

python run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--data_name $DATASET_NAME \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--use_adapter \
--non_sequential \
--adapter_names  $ADAPTER_NAMES\
--amp \
--cuda \
--num_workers 32 \
--max_seq_length 64 \
--batch_size 256 \
--lr 1e-04 \
--epochs 1 \
--save_step 2000
```
 </details>
 
### Train Downstream Tasks
To evaluate the model on a downstream task, you can go to the task folder and see the *.sh file for an example. For example, the following command is used to train a model on pubmedqa dataset over different shuffle_rates.
<details>
<summary>Click to expand!</summary>
  
```shell
MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TOKENIZER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
ADAPTER_NAMES="entity_predict"
PARTITION=20
shuffle_rates=(0.10 0.20 0.40 0.80 1.00)

for shuffle_rate in ${shuffle_rates[*]}; do
    python run_pretrain.py \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --n_partition $PARTITION \
    --use_adapter \
    --non_sequential \
    --adapter_names  $ADAPTER_NAMES\
    --amp \
    --cuda \
    --shuffle_rate $shuffle_rate \
    --num_workers 32 \
    --max_seq_length 64 \
    --batch_size 256 \
    --bi_direction \
    --lr 1e-04 \
    --epochs 2 \
    --save_step 2000
done
```
</details>
 
## Hyper-parameters

### Pre-train
<details>
<summary>Click to expand!</summary>
  
|Parameter|Value|
|------|------|
|lr|1e-04|
|epoch|1-2|
|batch_size|256|
|max_seq_length|64|
  </details>


### BioASQ7b,BioASQ8b,PubMedQA
<details>
<summary>Click to expand!</summary>
  
|Parameter|Value|
|------|------|
|lr|1e-05|
|epoch|25|
|patient|5|
|batch_size|8|
|max_seq_length|512|
|repeat_run|10|
</details>

### MedQA
<details>
<summary>Click to expand!</summary>
  
|Parameter|Value|
|------|------|
|lr|1e-05,2e-05|
|epoch|25|
|patient|5|
|batch_size|12|
|max_seq_length|512|
|repeat_run|3|
|temperature|1|
</details>

### MedNLI
<details>
<summary>Click to expand!</summary>
  
|Parameter|Value|
|------|------|
|lr|1e-05|
|epoch|25|
|patient|5|
|batch_size|16|
|max_seq_length|256|
|repeat_run|3|
|temperature|-15,-10,1|
</details>

### HoC
<details>
<summary>Click to expand!</summary>
  
|Parameter|Value|
|------|------|
|lr|1e-05,3e-05|
|epoch|25|
|patient|5|
|batch_size|16,32|
|max_seq_length|256|
|repeat_run|5|
|temperature|1|
</details>

### If you find our paper and resources useful, please kindly cite our paper:
```latex
@inproceedings{meng2021mixture,
  title={Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT},
  author={Meng, Zaiqiao and Liu, Fangyu and Clark, Thomas and Shareghi, Ehsan and Collier, Nigel},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={4672--4681},
  year={2021}
}
```

### Contact
If you have any questions, feel free to contact me via (zm324@cam.ac.uk).
