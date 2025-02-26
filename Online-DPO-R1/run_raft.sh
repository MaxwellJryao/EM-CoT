source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"


# Base paths and settings
initial_model="Qwen/Qwen2.5-Math-7B"
base_path="data/raft_numina_rule_reward"
mkdir $base_path
iteration_prefix="Train"
best_of_k=8
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
NUM_GPUS=$my_world_size
dataset_size=50

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5

    conda activate vllm
    my_world_size=$my_world_size
    infer_model=$2
    prompt_dir=$3
    output_dir=$4
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python ./generation/gen_hf.py \
            --model_name_or_path $model_path \
            --dataset_name_or_path $jsonl_input \
            --output_dir $json_output \
            --K $best_of_k \
            --temperature 1.0 \
            --local_index $i \
            --dataset_size $dataset_size \
            --my_world_size $my_world_size &
    done
  
    wait # Ensure all inference processes finish
    
    # Merge the generated data
    python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.jsonl" --num_datasets $my_world_size
    
    # Perform reward labeling
    python reward_labeling.py --dataset_name_or_path "${output_dir}_data.jsonl" --output_dir $model_output

    # Prepare the sft data for raft
    python raft/prepare_sft_data.py --data_path $model_output
   
    conda activate sft

    # CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") WANDB_MODE=offline accelerate launch --config_file ./configs/zero3.yaml --num_processes=$my_world_size --main_process_port=29501 dpo_iteration/run_dpo.py dpo_config.yaml
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") torchrun --nproc_per_node $my_world_size --master_port 20001 -m axolotl.cli.train raft/qwen.yaml \
        --deepspeed configs/deepspeed_stage3.json
}


# Main loop for iterations
for i in {1..1}
do
    iteration_name="Qwen_numina_raft${i}"
    jsonl_input="dsrtrain/numia_prompt"
    # jsonl_input="EleutherAI/hendrycks_math"
    json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}"
    model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward.json"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="Qwen_numina_raft${previous_iteration}"
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output
done

