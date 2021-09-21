DATASET="Hoc"
MODEL_DIR="model_dir"
DATA_DIR="data_dir/BLURB/data/HoC/"
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MODEL="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_20210309_101949_adapter"
T=1
LR=2e-5
BATCH_SIZE=32
TRAIN_MODE="fusion"

python eval_hoc.py \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--model $MODEL  \
--max_seq_length 128   \
--batch_size $BATCH_SIZE \
--lr $LR   \
--pretrain_epoch 0 \
--epochs 20 \
--repeat_runs 5 \
--temperature $T \
--cuda