DATASET="BioAsq8b"
MODEL_DIR="../../../../model_dir/"
DATA_DIR="../../../../../data/med_qa/bioasq8b/" 
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MODEL="PubMedBERT_S20Rel"
T=1
LR=1e-5
TRAIN_MODE="fusion"
    
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