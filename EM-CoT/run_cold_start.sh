CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,8,9 torchrun --nproc_per_node 8 --master_port 20003 -m axolotl.cli.train sft/qwen_cold_start.yaml \
        --deepspeed configs/deepspeed_stage3_offload.json