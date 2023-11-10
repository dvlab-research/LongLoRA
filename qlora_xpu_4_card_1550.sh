export CCL_ZE_IPC_EXCHANGE=sockets
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=28 # adjust this to 1/4 of total physical cores
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone \
         --nnodes=1 \
         --nproc-per-node 8 \
        supervised-fine-tune-qlora-debug.py  \
        --model_name_or_path "/home/wangruonan/Llama-2-7b-chat-hf" \
        --bf16 True \
        --output_dir "./qlora"       \
        --model_max_length 8096 \
        --use_flash_attn False \
        --data_path LongAlpaca-12k.json \
        --low_rank_training True \
        --num_train_epochs 3  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 2     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 500     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --gradient_checkpointing True \
        --logging_steps 1     \
        --tf32 False