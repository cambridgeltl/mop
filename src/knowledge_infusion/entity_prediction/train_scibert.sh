MODEL="allenai/scibert_scivocab_uncased"
TOKENIZER="allenai/scibert_scivocab_uncased"
INPUT_DIR="kg_dir"
OUTPUT_DIR="model_dir"
DATASET_NAME="snomed_ro"
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
--epochs 2 \
--save_step 2000