DATASET="MEDQA"
MODEL_DIR=model_dir
DATA_DIR="data_dir/MEDQA/data_clean/questions/US/4_options/"
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MODEL="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_20210328_163251_adapter"
T=1
LR=1e-5
BATCH_SIZE=12
TRAIN_MODE="fusion"
python run_medqa.py \
    --train_mode $TRAIN_MODE \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR  \
    --base_model $BASE_MODEL \
    --tokenizer $BASE_MODEL  \
    --model $MODEL  \
    --max_seq_length 512   \
    --batch_size $BATCH_SIZE \
    --lr $LR   \
    --repeat_runs 2 \
    --pretrain_epoch 0 \
    --epochs 25 \
    --temperature $T \
    --cuda