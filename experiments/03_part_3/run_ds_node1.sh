export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth
OUTPUT_DIR=/data/he.yan/experiments/alpaca-70b
WANDB_PROJECT="llama3_alpaca"
MODEL_NAME="meta-llama/Meta-Llama-3-70B"
MAX_SEQ_LEN=2048
EVAL_STRATEGY="no"
SAVE_STRATEGY="epoch"
DATA_DIR="./alpaca-cleaned-template-512"

accelerate launch \
    --config_file ./ds_node1_config.yaml \
    ../exp01/train.py \
    --seed 100 \
    --model_name $MODEL_NAME \
    --dataset_name $DATA_DIR \
    --chat_template_format "none" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train" \
    --max_seq_length $MAX_SEQ_LEN \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --log_level "info" \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_strategy "steps" \
    --eval_strategy $EVAL_STRATEGY \
    --save_strategy $SAVE_STRATEGY \
    --bf16 True \
    --packing True \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataset_text_field "content" \
    --use_flash_attn True \
    --ddp_timeout 5400 \
    --optim paged_adamw_32bit \
    # --push_to_hub \
    # --hub_private_repo True \
    # --hub_strategy "every_save"
