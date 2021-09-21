Dataset="PubMedQA"
MODEL_DIR=model_dir
BASE_MODEL="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DATA_DIR="data_dir/BLURB/data/pubmedqa/data/"
MODEL=BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_20210309_101949_adapter
LR=1e-5
EPOCH=0
T=1
BATCH_SIZE=8
SEQ_LENGTH=512

python eval_pubmedqa.py \
    --data_dir $DATA_DIR   \
    --model_dir $MODEL_DIR \
    --tokenizer $BASE_MODEL   \
    --base_model $BASE_MODEL \
    --cuda \
    --temperature $T \
    --model $Model   \
    --train_ratio $TRAIN_RATIO \
    --pretrain_epoch $EPOCH \
    --max_seq_length $SEQ_LENGTH   \
    --batch_size $BATCH_SIZE \
    --lr $LR   \
    --epochs 25