DATASET="BioAsq"
MODEL_DIR=model
DATA_DIR=/home/zm324/workspace/data/med_qa/bioasq8b/
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MODEL="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_20210309_101949_adapter"
Ts=(-2 -3 -5 -10 -15 0.25 0.5 0.75 0.999)
LR=1e-5
TRAIN_MODE="fusion"
for T in ${Ts[*]}; do
    python eval_bioasq.py \
    --train_mode $TRAIN_MODE \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR  \
    --base_model $BASE_MODEL \
    --tokenizer $BASE_MODEL  \
    --model $MODEL  \
    --max_seq_length 512   \
    --batch_size 8 \
    --lr $LR   \
    --pretrain_epoch 0 \
    --epochs 25 \
    --temperature $T \
    --cuda
done

MODEL="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_top20_20210219_081903_adapter"
for T in ${Ts[*]}; do
    
    python eval_bioasq.py \
    --train_mode $TRAIN_MODE \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR  \
    --base_model $BASE_MODEL \
    --tokenizer $BASE_MODEL  \
    --model $MODEL  \
    --max_seq_length 512   \
    --batch_size 8 \
    --lr $LR   \
    --pretrain_epoch 0 \
    --epochs 25 \
    --temperature $T \
    --cuda
done