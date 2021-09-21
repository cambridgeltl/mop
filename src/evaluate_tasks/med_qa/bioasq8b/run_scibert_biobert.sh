DATASET="BioAsq"
MODEL_DIR=model_dir
DATA_DIR=data_dir/bioasq8b/

BASE_MODEL="dmis-lab/biobert-v1.1"
MODEL="biobert-v1.1_snomed_ro_20210329_142151_adapter" \
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

BASE_MODEL="allenai/scibert_scivocab_cased"
MODEL="scibert_scivocab_uncased_snomed_ro_20210330_103845_adapter"
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