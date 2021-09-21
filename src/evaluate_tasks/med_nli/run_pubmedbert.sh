DATASET="MEDNLI"
MODEL_DIR="model_dir"
DATA_DIR="data_dir/blue/data/mednli/"
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MODEL="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_20210309_101949_adapter"
LR=1e-5
TRAIN_MODE="fusion"
python eval_nli.py \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--model $MODEL  \
--max_seq_length 256   \
--batch_size 16 \
--lr $LR   \
--repeat_runs 3 \
--pretrain_epoch 0 \
--epochs 25 \
--temperature $T \
--cuda