export OMP_NUM_THREADS=28

python supervised-fine-tune-qlora.py  \
        --model_name_or_path "/home/wangruonan/Llama-2-7b-chat-hf" \
        --bf16 True \
        --output_dir "./qlora"       \
        --model_max_length 16384 \
        --use_flash_attn False \
        --data_path LongAlpaca-12k.json \
        --low_rank_training True \
        --num_train_epochs 3  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 500     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --gradient_checkpointing False \
        --logging_steps 1     \
        --tf32 False