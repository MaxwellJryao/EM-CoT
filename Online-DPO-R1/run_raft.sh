source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"


# Base paths and settings
initial_model="Qwen/Qwen2.5-Math-7B"
base_path="data/raft_numina_rule_reward"
mkdir -p $base_path
iteration_prefix="Train"
best_of_k=8
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
NUM_GPUS=$my_world_size
dataset_start=0
dataset_end=10000

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5
    local iteration_num=$6
    local suffix=$7

    if [ $iteration_num -eq 1 ]; then
        data_shuffle_seed=42
    else
        data_shuffle_seed=$iteration_num
    fi

    # conda activate vllm
    # my_world_size=$my_world_size
    # infer_model=$2
    # prompt_dir=$3
    # output_dir=$4
    # for i in $(seq 0 $((NUM_GPUS - 1))); do
    #     CUDA_VISIBLE_DEVICES=${GPUS[$i]} python ./generation/gen_hf.py \
    #         --model_name_or_path $model_path \
    #         --dataset_name_or_path $jsonl_input \
    #         --output_dir $json_output \
    #         --K $best_of_k \
    #         --temperature 1.0 \
    #         --local_index $i \
    #         --dataset_end $dataset_end \
    #         --dataset_start $dataset_start \
    #         --data_shuffle_seed $iteration_num \
    #         --my_world_size $my_world_size &
    # done
  
    # wait # Ensure all inference processes finish
    
    # Merge the generated data
    # python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.jsonl" --num_datasets $my_world_size
    
    # Perform reward labeling
    # python reward_labeling.py --dataset_name_or_path "${output_dir}_data.jsonl" --output_dir $model_output --iter=$iteration_num

    # Prepare the sft data for raft
    # python raft/prepare_sft_data.py --data_path $model_output --start=$dataset_start --end=$dataset_end --iter=$iteration_num
   
    conda activate sft

    cat <<EOT > raft/qwen.yaml
base_model: $model_path
trust_remote_code: false

load_in_8bit: false
load_in_4bit: false
strict: false

chat_template: qwen_25
datasets:
  - path: data/raft_train_iter${iteration_num}_${dataset_start}_${dataset_end}
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
output_dir: ./outputs/${iteration}

sequence_len: 8192
sample_packing: true
eval_sample_packing: true
pad_to_sequence_len: true

# wandb_project: huggingface
# wandb_entity: zzzzzaa
# wandb_watch:
# wandb_name: qwen_test
# wandb_log_model:

gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 1
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

    # CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") WANDB_MODE=offline accelerate launch --config_file ./configs/zero3.yaml --num_processes=$my_world_size --main_process_port=29501 dpo_iteration/run_dpo.py dpo_config.yaml
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") torchrun --nproc_per_node $my_world_size --master_port 20001 -m axolotl.cli.train raft/qwen.yaml \
        --deepspeed configs/deepspeed_stage3.json
    # CUDA_VISIBLE_DEVICES=0 axolotl preprocess raft/qwen.yaml --debug
}


# Main loop for iterations
for i in {1..1}
do
    suffix="orig_eos"
    if [ -z $suffix ]; then
        echo "No suffix"
        iteration_name="Qwen_numina_raft${i}"
        json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_${dataset_start}-${dataset_end}"
        model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward_${dataset_start}-${dataset_end}.json"        
    else
        echo "Suffix: $suffix"
        iteration_name="Qwen_numina_raft${i}_${suffix}"
        json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_${suffix}_${dataset_start}-${dataset_end}"
        model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_${suffix}_reward_${dataset_start}-${dataset_end}.json"
    fi
    jsonl_input="dsrtrain/numia_prompt"
    # jsonl_input="EleutherAI/hendrycks_math"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        if [ -z $suffix ]; then
            model_path="outputs/Qwen_numina_raft${previous_iteration}"
        else
            model_path="outputs/Qwen_numina_raft${previous_iteration}_${suffix}"
        fi
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output $i $suffix
done

