set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

data=numina_math_flat_em
project_name=verl-math
algorithm=raft
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/Qwen2.5-Math-1.5B
policy_loss=vanilla # vanilla, ppo (importance sample + clipping)
experiment_name=${model}-${algorithm}-${policy_loss}-${data}-iter1
GPUS=(1 2 3 4 5 6 7 8)
my_world_size=${#GPUS[@]}
total_epochs=3

math_train_path=./em/data/Qwen2.5-Math-1.5B/numina_math_10k_1/data_1/train.parquet
math_test_path=./data/math500/test.parquet 

train_files="['$math_train_path']"
test_files="['$math_test_path']"

# run_iteration() {
#     local start_model=$1

start_model=$model_name_or_path
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$start_model" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$my_world_size \
    +trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs $@

# }

# for i in $(seq 1 $total_epochs); do
#     if [ $i -eq 1 ]; then
#         start_model=$model_name_or_path
#     else
#         # first merge model into safetensors format
#         python scripts/model_merger.py --local_dir=checkpoints/${project_name}/${experiment_name}/global_step_$((5*(i-1)))/actor
#         start_model=checkpoints/${project_name}/${experiment_name}/global_step_$((5*(i-1)))/actor/huggingface
#     fi
#     run_iteration $start_model
# done