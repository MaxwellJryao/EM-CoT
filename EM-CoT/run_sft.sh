GPUS=(1 2 3 5 6 7 8 9)
my_world_size=${#GPUS[@]}

source ~/.bashrc
conda activate sft

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") torchrun --nproc_per_node $my_world_size --master_port 20001 -m axolotl.cli.train sft/qwen.yaml \
    --deepspeed configs/deepspeed_stage3.json