Dataset="PubMedQA"
MODEL_DIR="../../../../model_dir/"
DATA_DIR="../../../../../data/BLURB/data/pubmedqa/" 
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MODEL="PubMedBERT_S20Rel"
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
    --model $MODEL   \
    --pretrain_epoch $EPOCH \
    --max_seq_length $SEQ_LENGTH   \
    --batch_size $BATCH_SIZE \
    --lr $LR   \
    --epochs 25