MODEL="facebook/bart-base"
TOKENIZER="facebook/bart-base"
INPUT_DIR="/home/linry/bart--prompt-closed-qa/data"
OUTPUT_DIR="/home/linry/bart--prompt-closed-qa/checkpoints"
DATASET_NAME="wikidata5m"
ADAPTER_NAMES="entity_predict"
ADAPTER_TYPE="PrefixTuningConfig"
PREFIX_LENGTH=100
PROMPT_LENGTH=64
PARTITION=20
TRIPLE_PER_RELATION=5000
EPOCH=5
LR=1e-04

# python ./src/relation_prompt/run_pretrain.py \
# --model $MODEL \
# --tokenizer $TOKENIZER \
# --input_dir $INPUT_DIR \
# --output_dir $OUTPUT_DIR \
# --n_partition $PARTITION \
# --triple_per_relation $TRIPLE_PER_RELATION \
# --adapter_names  $ADAPTER_NAMES \
# --adapter_type $ADAPTER_TYPE \
# --prefix_length $PREFIX_LENGTH \
# --prompt_length $PROMPT_LENGTH \
# --use_adapter \
# --cuda \
# --num_workers 16 \
# --max_seq_length 64 \
# --batch_size 16 \
# --lr $LR \
# --epochs $EPOCH

python ./src/relation_prompt/run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--triple_per_relation $TRIPLE_PER_RELATION \
--adapter_names  $ADAPTER_NAMES \
--adapter_type $ADAPTER_TYPE \
--prefix_length $PREFIX_LENGTH \
--prompt_length $PROMPT_LENGTH \
--use_adapter \
--cuda \
--num_workers 16 \
--max_seq_length 64 \
--batch_size 16 \
--lr $LR \
--epochs $EPOCH \
--use_prompt