MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TOKENIZER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
INPUT_DIR="kg_dir"
OUTPUT_DIR="model_dir"
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