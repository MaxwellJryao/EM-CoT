source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

export TOKENIZERS_PARALLELISM=false

initial_model="Qwen/Qwen2.5-Math-1.5B-Instruct"
act_params="embed_tokens" # embed_tokens for Qwen-1.5B, lm_head for Qwen-7B
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
model_prefix="Qwen1.5B-Inst"
data_start=0
data_end=5000

run_iteration() {
    local iteration_num=$1
    local data_path=$2
    local model_name_or_path=$3

    conda activate vllm
    for i in $(seq 0 $((my_world_size - 1))); do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_1_collect_data.py --local_index $i --world_size $my_world_size \
            --model_name_or_path $model_name_or_path --iter $iteration_num --data_path $data_path \
            --model_prefix=$model_prefix --end=$data_end &
    done

    wait

    for i in $(seq 0 $((my_world_size - 1))); do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_2_calc_sample_size.py --local_index $i --iter $iteration_num \
          --model_name_or_path=$model_name_or_path --act_params=$act_params --model_prefix=$model_prefix \
          --end=$data_end &
    done

    wait

    for i in $(seq 0 $((my_world_size - 1))); do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_2_sample.py --local_index $i --iter $iteration_num \
          --model_name_or_path=$model_name_or_path --model_prefix=$model_prefix --end=$data_end &
    done

    wait

    python stage_2_merge_data.py --iter $iteration_num --num_collect_files $my_world_size --model_prefix=$model_prefix \
      --train_size=$data_end

    conda activate sft

    cat <<EOT > sft/qwen.yaml
base_model: $model_name_or_path
trust_remote_code: false

load_in_8bit: false
load_in_4bit: false
strict: false

chat_template: qwen_25
datasets:
  - path: data/${model_prefix}/data_${iteration_num}/train_data
    type: chat_template
    field_messages: conversations
    message_field_role: role
    message_field_content: content
    roles:
      user: ["human", "user"]
      assistant: ["gpt", "assistant", "ai"]
      system: ["system"]

dataset_prepared_path:
val_set_size: 0.0
output_dir: /shared/storage-01/jiarui14/EM-CoT/EM-CoT/outputs/${model_prefix}_sft_${iteration_num}

sequence_len: 8192
sample_packing: true
eval_sample_packing: true
pad_to_sequence_len: true

# wandb_project: huggingface
# wandb_entity: zzzzzaa
# wandb_watch:
# wandb_name: qwen_test
# wandb_log_model:

gradient_accumulation_steps: $((64 / my_world_size))
micro_batch_size: 1
num_epochs: 3
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 1e-5

train_on_inputs: false
group_by_length: false
bf16: auto
fp16: false
tf32: true

gradient_checkpointing: false
gradient_checkpointing_kwargs:
  use_reentrant: false
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_ratio: 0.05
saves_per_epoch: 1
evals_per_epoch: 0
debug:
weight_decay: 0.01
fsdp:
fsdp_config:
# special_tokens:
#   bos_token: "<|im_start|>"
#   eos_token: "<|im_end|>"
#   pad_token: "<|endoftext|>"


plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true
liger_layer_norm: true
liger_fused_linear_cross_entropy: true
EOT

    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") torchrun --nproc_per_node $my_world_size --master_port 20002 -m axolotl.cli.train sft/qwen.yaml \
        --deepspeed configs/deepspeed_stage3.json
}

for i in {1..1}
do
    mkdir -p data/${model_prefix}/data_${i}

    if [ $i -eq 1 ]; then
        model_name_or_path=$initial_model
    else
        previoud_iteration=$((i-1))
        model_name_or_path="/shared/storage-01/jiarui14/EM-CoT/EM-CoT/outputs/${model_prefix}_sft_${previoud_iteration}"
    fi

    data_path="FlippyDora/raft${i}_train_numia_prompt_0-10000"

    run_iteration $i $data_path $model_name_or_path
done


# model_name_or_path="outputs/qwen_sft"
# iter=2
# data_path="FlippyDora/raft2_train_numia_prompt_0-10000"

# CUDA_VISIBLE_DEVICES=8 python stage_1_collect_data.py --local_index=0 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=1 python stage_1_collect_data.py --local_index=1 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=2 python stage_1_collect_data.py --local_index=2 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=3 python stage_1_collect_data.py --local_index=3 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=4 python stage_1_collect_data.py --local_index=4 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=5 python stage_1_collect_data.py --local_index=5 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=6 python stage_1_collect_data.py --local_index=6 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &
# CUDA_VISIBLE_DEVICES=7 python stage_1_collect_data.py --local_index=7 --world_size=8 --model_name_or_path=$model_name_or_path --iter=$iter --data_path=$data_path &

# CUDA_VISIBLE_DEVICES=0 python stage_2_calc_sample_size.py --local_index=0 &
# CUDA_VISIBLE_DEVICES=1 python stage_2_calc_sample_size.py --local_index=1 &
# CUDA_VISIBLE_DEVICES=2 python stage_2_calc_sample_size.py --local_index=2 &
# CUDA_VISIBLE_DEVICES=3 python stage_2_calc_sample_size.py --local_index=3 &
# CUDA_VISIBLE_DEVICES=4 python stage_2_calc_sample_size.py --local_index=4 &
# CUDA_VISIBLE_DEVICES=5 python stage_2_calc_sample_size.py --local_index=5 &
# CUDA_VISIBLE_DEVICES=6 python stage_2_calc_sample_size.py --local_index=6 &
# CUDA_VISIBLE_DEVICES=7 python stage_2_calc_sample_size.py --local_index=7 &

# CUDA_VISIBLE_DEVICES=8 python stage_2_sample.py --local_index=0 &
# CUDA_VISIBLE_DEVICES=1 python stage_2_sample.py --local_index=1 &
# CUDA_VISIBLE_DEVICES=2 python stage_2_sample.py --local_index=2 &
# CUDA_VISIBLE_DEVICES=3 python stage_2_sample.py --local_index=3 &
# CUDA_VISIBLE_DEVICES=4 python stage_2_sample.py --local_index=4 &
# CUDA_VISIBLE_DEVICES=5 python stage_2_sample.py --local_index=5 &
# CUDA_VISIBLE_DEVICES=6 python stage_2_sample.py --local_index=6 &
# CUDA_VISIBLE_DEVICES=7 python stage_2_sample.py --local_index=7 &