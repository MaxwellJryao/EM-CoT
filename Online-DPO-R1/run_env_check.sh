source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"


# Base paths and settings
initial_model="Qwen/Qwen2.5-Math-7B"
base_path="./iter_dpo_numina_rule_reward"
mkdir $base_path
iteration_prefix="Test"
best_of_k=4
my_world_size=8
NUM_GPUS=$my_world_size


# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5

    conda activate vllm
    infer_model=$2
    prompt_dir=$3
    output_dir=$4
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$i python ./generation/gen_hf.py \
            --model_name_or_path $model_path \
            --dataset_name_or_path $jsonl_input \
            --output_dir $json_output \
            --K $best_of_k \
            --temperature 1.0 \
            --local_index $i \
            --my_world_size $my_world_size &
    done
  
    wait # Ensure all inference processes finish
    
    # Merge the generated data
    python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.jsonl" --num_datasets 8
    
    # Perform reward labeling
    python reward_labeling.py --dataset_name_or_path "${output_dir}_data.jsonl" --output_dir $model_output
    conda activate rlhflow
    cat <<EOT > dpo_config.yaml
run_name: $iteration
output_dir: $iteration
model_name_or_path: $model_path
ref_model: $model_path
learning_rate: 5.0e-7
num_train_epochs: 2
logging_steps: 2
gradient_checkpointing: true
do_train: true
do_eval: true
eval_steps: 10000
choose_type: max_min
train_dir: $model_output
eval_dir: $model_output
loss_type: sigmoid
lr_scheduler_type: cosine
max_length: 4096
max_prompt_length: 1000
eval_strategy: steps
bf16: true
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
report_to: wandb
label_smoothing: 0.1
eot_token: <|im_end|>
EOT

    accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_dpo.py dpo_config.yaml
}



iteration_name="Qwen_numina_initial_test"
jsonl_input="RLHFlow/numia_prompt_dpo_test"
json_output="${base_path}/${iteration_prefix}_${iteration_name}"
model_output="${base_path}/${iteration_prefix}_${iteration_name}_reward.json"
model_path=$initial_model


run_iteration "$iteration_name" "$model_path" "$jsonl_input" "$json_output" "$model_output"

