source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

initial_model="/shared/storage-01/jiarui14/EM-CoT/verl/checkpoints/verl-math/Qwen2.5-Math-1.5B-raft-vanilla-numina_math_flat_em-iter1/global_step_210/actor/huggingface"
act_params="embed_tokens" # embed_tokens for Qwen-1.5B, lm_head for Qwen-7B
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
model_prefix="Qwen2.5-Math-1.5B"
data_start=0
data_end=200000
stage_1_samples_per_prompt=8
stage_2_samples=$((stage_1_samples_per_prompt*(data_end-data_start)))
train_size=200000
alpha=1e-3
beta=2.0
system_prompt="qwen25-math-cot" # "qwen25-math-cot", "hendrydong-longcot"
suffix="numina_math_10k_1"

i=2
iteration_num=$i
model_name_or_path=$initial_model
data_path=data/${model_prefix}/${suffix}/train_ds

mkdir -p data/${model_prefix}/${suffix}/data_${i}

conda activate vllm
for i in $(seq 0 $((my_world_size - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_1_collect_data.py --local_index $i --world_size $my_world_size \
        --model_name_or_path $model_name_or_path --iter $iteration_num --data_path $data_path \
        --model_prefix=$model_prefix --end=$data_end --suffix=$suffix --stage_1_samples=$stage_1_samples_per_prompt \
        --system_prompt=$system_prompt &
done

wait

for i in $(seq 0 $((my_world_size - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_2_calc_acceptRates_grads.py --local_index $i --iter $iteration_num \
        --model_name_or_path=$model_name_or_path --act_params=$act_params --model_prefix=$model_prefix \
        --end=$data_end --suffix=$suffix --num_collect_files=$my_world_size --stage_1_samples=$stage_1_samples_per_prompt \
        --system_prompt=$system_prompt &
done

wait

python stage_2_calc_sample_size.py --num_collect_files=$my_world_size --suffix=$suffix --iter=$iteration_num \
    --model_prefix=$model_prefix --stage_2_samples=$stage_2_samples --alpha=$alpha --beta=$beta --stage_1_samples=$stage_1_samples_per_prompt

# for i in $(seq 0 $((my_world_size - 1))); do
#     CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_2_sample.py --local_index $i --iter $iteration_num \
#         --model_name_or_path=$model_name_or_path --model_prefix=$model_prefix --end=$data_end --suffix=$suffix \
#         --system_prompt=$system_prompt &
# done

# wait

python stage_2_merge_data.py --iter $iteration_num --num_collect_files $my_world_size --model_prefix=$model_prefix \
    --train_size=$train_size --suffix=$suffix --system_prompt=$system_prompt