GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") WANDB_MODE=offline accelerate launch --config_file ./configs/zero3.yaml \
    --num_processes=$my_world_size --main_process_port=29502 dpo/run_dpo.py dpo/dpo_config.yaml