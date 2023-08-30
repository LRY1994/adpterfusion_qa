DATASET="WebQuestion"
DATA_DIR="/home/linry/bart--prompt-closed-qa/data/WebQuestion/splitted/"
MODEL_DIR_TRUE="/home/linry/bart--prompt-closed-qa/checkpoints/bart-base_PrefixTuningConfig_20_5000_100_64_True"
MODEL_DIR_FALSE="/home/linry/bart--prompt-closed-qa/checkpoints/bart-base_PrefixTuningConfig_20_5000_100_64_False"
BASE_MODEL="facebook/bart-base"
T=1
LR=1e-5
TRAIN_MODE_FUSION="fusion"
TRAIN_MODE_BASE="base"
OUTPUT_DIR="output"
PARTITION=5
TRAIN_BATCH_SIZE=8
PRE_EPOCH=4
EPOCH=1

# FUSION NO PROMPT pretrain_epoch 
# python ./src/evaluation_BART/eval_question.py \
# --dataset $DATASET \
# --train_mode $TRAIN_MODE_FUSION \
# --model_dir $MODEL_DIR_FALSE \
# --data_dir $DATA_DIR  \
# --base_model $BASE_MODEL \
# --tokenizer $BASE_MODEL  \
# --adapter_num $PARTITION \
# --batch_size $TRAIN_BATCH_SIZE \
# --eval_batch_size $TRAIN_BATCH_SIZE \
# --max_input_length 64 \
# --max_output_length 64 \
# --learning_rate $LR   \
# --pretrain_epoch $PRE_EPOCH \
# --epochs $EPOCH \
# --repeat_runs 1 \
# --temperature $T \
# --output_dir $OUTPUT_DIR \
# --gradient_accumulation_steps 4 \
# --cuda 


# FUSION PROMPT pretrain_epoch 
python ./src/evaluation_BART/eval_question.py \
--dataset $DATASET \
--train_mode $TRAIN_MODE_FUSION \
--model_dir $MODEL_DIR_TRUE \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--adapter_num $PARTITION \
--batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $TRAIN_BATCH_SIZE \
--max_input_length 64 \
--max_output_length 64 \
--learning_rate $LR   \
--pretrain_epoch $PRE_EPOCH \
--epochs $EPOCH \
--repeat_runs 1 \
--temperature $T \
--output_dir $OUTPUT_DIR \
--gradient_accumulation_steps 4 \
--cuda 



# #bash src/evaluation_BART/run_eval_wq.sh
# #sudo wg-quick up tw
# # git remote set-url origin https://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@github.com/LRY1994/knowledge-infusion.git/

