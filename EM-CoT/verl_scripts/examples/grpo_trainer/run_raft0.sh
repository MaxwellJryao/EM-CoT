set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

data=numina_math
project_name=verl-math
algorithm=raft
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/Qwen2.5-Math-1.5B
policy_loss=vanilla # vanilla, ppo (importance sample + clipping)
experiment_name=${model}-${algorithm}-${policy_loss}-${data}
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}

math_train_path=./data/$data/train.parquet
math_test_path=./data/math500/test.parquet 

train_files="['$math_train_path']"
test_files="['$math_test_path']"

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$model_name_or_path \
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
    actor_rollout_ref.rollout.n=4 \
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
    trainer.total_epochs=15 $@

